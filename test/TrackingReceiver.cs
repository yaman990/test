using System;
using System.Collections.Concurrent;
using System.Collections.Generic;
using System.Globalization;
using System.Net;
using System.Net.Sockets;
using System.Text;
using System.Threading;
using Newtonsoft.Json.Linq;
using UnityEngine;

public class TrackingReceiver : MonoBehaviour
{
    [Header("UDP")]
    public string listenHost = "127.0.0.1";
    public int listenPort = 5053;

    [Header("Debug")]
    public bool debugLogs = false;
    public float staleAfterSeconds = 0.5f;

    public class LandmarkData
    {
        public float x;
        public float y;
        public float z;
        /// <summary>MediaPipe visibility 0..1; 1 when omitted (old packets).</summary>
        public float visibility = 1f;
    }

    public class FaceData
    {
        public float cx;
        public float cy;
        public float w;
        public float h;
        public List<LandmarkData> landmarks = new List<LandmarkData>();
    }

    /// <summary>
    /// Arm pole vectors from Python: unit normals of (shoulder, elbow, wrist) triangles in world meters.
    /// Same joint order as compact <c>poses</c> / MediaPipe BlazePose (left arm then right).
    /// </summary>
    public struct ElbowPolePair
    {
        public Vector3 left;
        public Vector3 right;
    }

    public class TrackingFrameData
    {
        /// <summary>Python packet schema: 1 = legacy, 2+ = includes world_poses and extras.</summary>
        public int protocolVersion = 1;
        public long timestampMs;
        public int seq;
        public int cameraW;
        public int cameraH;
        /// <summary>True if Python sent mirror_scale (tracker computes distance / mirror sizing).</summary>
        public bool hasMirrorHint;
        public float mirrorScale;
        public float mirrorZPull;
        public List<FaceData> faces = new List<FaceData>();
        /// <summary>Normalized image landmarks (0..1) + z depth hint; compact 13 joints unless full pose.</summary>
        public List<List<LandmarkData>> poses = new List<List<LandmarkData>>();
        /// <summary>True when <c>world_poses</c> was parsed (meters, hip origin, One-Euro smoothed on Python side).</summary>
        public bool hasWorldPoses;
        /// <summary>Per detected person: 13 world-space points in meters, same joint order as compact <c>poses</c>.</summary>
        public List<List<Vector3>> worldPoses = new List<List<Vector3>>();
        /// <summary>Per person: visibility 0..1 for each of the 13 compact joints (image-space scores from MediaPipe).</summary>
        public List<List<float>> poseVisibility = new List<List<float>>();
        /// <summary>Mean visibility of the 13 compact joints for person 0; use for show/hide without heuristics.</summary>
        public float personVisibility = 1f;
        /// <summary>Python gate: mean joint visibility exceeded threshold; hide avatar when false.</summary>
        public bool isPresent = true;
        /// <summary>Rolling estimate of lowest ankle Y in world meters (foot / floor contact offset).</summary>
        public float floorY;
        /// <summary>Smoothed capture / service FPS from Python.</summary>
        public float trackerFps;
        /// <summary>Per person: left and right elbow pole (hinge facing) in world space.</summary>
        public List<ElbowPolePair> elbowPoles = new List<ElbowPolePair>();
    }

    private UdpClient _udp;
    private Thread _recvThread;
    private volatile bool _running;
    private readonly ConcurrentQueue<TrackingFrameData> _queue = new ConcurrentQueue<TrackingFrameData>();

    public TrackingFrameData LatestFrame { get; private set; }
    public float LastPacketRealtime { get; private set; }

    public bool HasRecentData()
    {
        return (Time.realtimeSinceStartup - LastPacketRealtime) <= staleAfterSeconds;
    }

    private void Start()
    {
        StartReceiver();
    }

    private void Update()
    {
        while (_queue.TryDequeue(out var frame))
        {
            LatestFrame = frame;
            LastPacketRealtime = Time.realtimeSinceStartup;
        }
    }

    private void OnDestroy()
    {
        StopReceiver();
    }

    private void OnApplicationQuit()
    {
        StopReceiver();
    }

    private void StartReceiver()
    {
        try
        {
            var ip = IPAddress.Parse(listenHost);
            _udp = new UdpClient(new IPEndPoint(ip, listenPort));
            _udp.Client.ReceiveTimeout = 1000;
            _running = true;

            _recvThread = new Thread(ReceiveLoop);
            _recvThread.IsBackground = true;
            _recvThread.Start();

            if (debugLogs)
            {
                Debug.Log($"[TrackingReceiver] Listening on udp://{listenHost}:{listenPort}");
            }
        }
        catch (Exception ex)
        {
            Debug.LogError($"[TrackingReceiver] Failed to start: {ex.Message}");
        }
    }

    private void StopReceiver()
    {
        _running = false;

        try { _udp?.Close(); } catch { }
        _udp = null;

        if (_recvThread != null && _recvThread.IsAlive)
        {
            try { _recvThread.Join(200); } catch { }
        }
        _recvThread = null;
    }

    private void ReceiveLoop()
    {
        var remote = new IPEndPoint(IPAddress.Any, 0);

        while (_running)
        {
            try
            {
                if (_udp == null) break;
                var bytes = _udp.Receive(ref remote);
                if (bytes == null || bytes.Length == 0) continue;

                var json = Encoding.UTF8.GetString(bytes);
                var frame = ParseFrame(json);
                if (frame != null)
                {
                    _queue.Enqueue(frame);
                }
            }
            catch (SocketException se)
            {
                // 10060 timeout is normal when no packets arrive during timeout window.
                if (se.SocketErrorCode != SocketError.TimedOut && debugLogs)
                {
                    Debug.LogWarning($"[TrackingReceiver] Socket warning: {se.Message}");
                }
            }
            catch (ObjectDisposedException)
            {
                break;
            }
            catch (Exception ex)
            {
                if (debugLogs)
                {
                    Debug.LogWarning($"[TrackingReceiver] Parse/receive error: {ex.Message}");
                }
            }
        }
    }

    private TrackingFrameData ParseFrame(string json)
    {
        var root = JObject.Parse(json);
        if ((string)root["type"] != "tracking_frame") return null;

        var frame = new TrackingFrameData
        {
            protocolVersion = (int?)root["version"] ?? 1,
            timestampMs = (long?)root["timestamp_ms"] ?? 0,
            seq = (int?)root["seq"] ?? -1,
            cameraW = (int?)root["camera"]?["w"] ?? 0,
            cameraH = (int?)root["camera"]?["h"] ?? 0
        };

        var ms = root["mirror_scale"];
        if (ms != null && ms.Type != JTokenType.Null)
        {
            frame.hasMirrorHint = true;
            frame.mirrorScale = SafeJsonFloat(ms, 1f);
            frame.mirrorZPull = SafeJsonFloat(root["mirror_z_pull"], 0f);
        }

        var faces = root["faces"] as JArray;
        if (faces != null)
        {
            foreach (var f in faces)
            {
                var face = new FaceData
                {
                    cx = (float?)f["bbox"]?["cx"] ?? 0f,
                    cy = (float?)f["bbox"]?["cy"] ?? 0f,
                    w = (float?)f["bbox"]?["w"] ?? 0f,
                    h = (float?)f["bbox"]?["h"] ?? 0f
                };

                var lms = f["landmarks"] as JArray;
                if (lms != null)
                {
                    foreach (var lm in lms)
                    {
                        face.landmarks.Add(new LandmarkData
                        {
                            x = (float?)lm["x"] ?? 0f,
                            y = (float?)lm["y"] ?? 0f,
                            z = (float?)lm["z"] ?? 0f,
                            visibility = SafeJsonFloat(lm["v"], 1f)
                        });
                    }
                }
                frame.faces.Add(face);
            }
        }

        var poses = root["poses"] as JArray;
        if (poses != null)
        {
            foreach (var pose in poses)
            {
                var poseList = new List<LandmarkData>();
                var joints = pose as JArray;
                if (joints != null)
                {
                    foreach (var j in joints)
                    {
                        poseList.Add(new LandmarkData
                        {
                            x = (float?)j["x"] ?? 0f,
                            y = (float?)j["y"] ?? 0f,
                            z = (float?)j["z"] ?? 0f,
                            visibility = SafeJsonFloat(j["v"], 1f)
                        });
                    }
                }
                frame.poses.Add(poseList);
            }
        }

        ParseMetersPayload(root, frame);

        return frame;
    }

    /// <summary>
    /// Fills fields produced by tracker_service.py (version 2+): world_poses, pose_visibility, elbow_poles, etc.
    /// Safe no-op when keys are missing (older Python or face-only).
    /// </summary>
    private static void ParseMetersPayload(JObject root, TrackingFrameData frame)
    {
        var visTok = root["is_visible"];
        if (visTok != null && visTok.Type != JTokenType.Null)
            frame.personVisibility = SafeJsonFloat(visTok, 1f);

        var presTok = root["is_present"];
        if (presTok != null && presTok.Type != JTokenType.Null)
        {
            if (presTok.Type == JTokenType.Boolean)
                frame.isPresent = presTok.Value<bool>();
            else
                frame.isPresent = SafeJsonFloat(presTok, 1f) >= 0.5f;
        }

        frame.floorY = SafeJsonFloat(root["floor_y"], 0f);
        frame.trackerFps = SafeJsonFloat(root["fps"], 0f);

        var worldArr = root["world_poses"] as JArray;
        if (worldArr != null)
        {
            foreach (var person in worldArr)
            {
                var joints = person as JArray;
                if (joints == null) continue;
                var row = new List<Vector3>(joints.Count);
                foreach (var jt in joints)
                {
                    var xyz = jt as JArray;
                    if (xyz != null && xyz.Count >= 3)
                    {
                        row.Add(new Vector3(
                            SafeJsonFloat(xyz[0], 0f),
                            SafeJsonFloat(xyz[1], 0f),
                            SafeJsonFloat(xyz[2], 0f)));
                    }
                    else
                    {
                        var jo = jt as JObject;
                        if (jo != null)
                        {
                            row.Add(new Vector3(
                                SafeJsonFloat(jo["x"], 0f),
                                SafeJsonFloat(jo["y"], 0f),
                                SafeJsonFloat(jo["z"], 0f)));
                        }
                    }
                }
                if (row.Count > 0)
                    frame.worldPoses.Add(row);
            }
        }

        var pvArr = root["pose_visibility"] as JArray;
        if (pvArr != null)
        {
            foreach (var person in pvArr)
            {
                var vals = person as JArray;
                if (vals == null) continue;
                var row = new List<float>(vals.Count);
                foreach (var v in vals)
                    row.Add(SafeJsonFloat(v, 1f));
                frame.poseVisibility.Add(row);
            }
        }

        var elbowArr = root["elbow_poles"] as JArray;
        if (elbowArr != null)
        {
            foreach (var person in elbowArr)
            {
                var pair = person as JArray;
                if (pair == null || pair.Count < 2) continue;
                var left = pair[0] as JArray;
                var right = pair[1] as JArray;
                var ep = new ElbowPolePair
                {
                    left = ReadVec3(left),
                    right = ReadVec3(right)
                };
                frame.elbowPoles.Add(ep);
            }
        }

        frame.hasWorldPoses = frame.worldPoses.Count > 0;
    }

    private static Vector3 ReadVec3(JArray xyz)
    {
        if (xyz == null || xyz.Count < 3)
            return Vector3.zero;
        return new Vector3(
            SafeJsonFloat(xyz[0], 0f),
            SafeJsonFloat(xyz[1], 0f),
            SafeJsonFloat(xyz[2], 0f));
    }

    private static float SafeJsonFloat(JToken token, float fallback)
    {
        if (token == null || token.Type == JTokenType.Null) return fallback;
        try
        {
            return token.Value<float>();
        }
        catch
        {
            if (float.TryParse(token.ToString(), NumberStyles.Float, CultureInfo.InvariantCulture, out var f))
                return f;
            return fallback;
        }
    }
}
