using System.Collections.Generic;
using UnityEngine;

/// <summary>
/// Builds a simple stick-figure (empty joints + optional spheres) and drives it from
/// tracker_service.py compact pose: 13 joints in order:
/// 0 nose, 1 L shoulder, 2 R shoulder, 3 L elbow, 4 R elbow, 5 L wrist, 6 R wrist,
/// 7 L hip, 8 R hip, 9 L knee, 10 R knee, 11 L ankle, 12 R ankle.
/// </summary>
public class StickFigureDriver : MonoBehaviour
{
    public TrackingReceiver receiver;

    [Header("Build")]
    public bool autoBuildOnStart = true;
    public bool addJointSpheres = true;
    public float jointSphereScale = 0.04f;
    public Material jointMaterial;

    [Header("Layout (world space relative to this object)")]
    [Tooltip("Match Python/OpenCV overlay: X uses camera aspect (W/H) × vertical scale so proportions match the preview.")]
    public bool matchPreviewPixelAspect = true;
    [Tooltip("Used when Match preview aspect is off; otherwise vertical size only (horizontal comes from aspect).")]
    public float widthScale = 1.6f;
    [Tooltip("Vertical spread in world units (nose–feet); also sets scale when matching preview aspect.")]
    public float heightScale = 1.8f;
    [Tooltip("Depth from MediaPipe z")]
    public float depthScale = 0.8f;
    [Tooltip("Mirror like a selfie (flip X)")]
    public bool mirrorX = true;

    [Header("Video frame (only what the webcam sees)")]
    [Tooltip("Keep joints inside the image rectangle so the mesh matches the live frame")]
    public bool clampToVideoFrame = true;
    [Range(0f, 0.15f)]
    public float frameEdgeMargin = 0.02f;
    [Tooltip("Only used when Clamp is OFF. When Clamp is ON, bones always connect clamped joints (partial body at edges).")]
    public bool hideBonesWhenBothEndsOffScreen = false;
    [Tooltip("When Clamp is ON, spheres stay visible at frame edges. When OFF, hide spheres for landmarks outside frame")]
    public bool hideJointSpheresOutsideFrame = true;
    [Tooltip("Hide bones/joints MediaPipe marks low-confidence (stops phantom legs when lower body is off-camera).")]
    public bool hideLowVisibilityLimbs = true;
    [Tooltip("Hide the whole stick figure when tracked body leaves camera frame.")]
    public bool hideWhenSubjectOffscreen = true;
    [Range(0f, 1f)]
    [Tooltip("Bone or joint shown only if both endpoints (bone) or point (sphere) are at or above this visibility.")]
    public float minLimbVisibility = 0.35f;

    [Header("Mirror distance (closer = bigger, like a real mirror)")]
    [Tooltip("Scale whole figure from how large your face appears on screen")]
    public bool mirrorDistanceSizing = true;
    [Tooltip("Use mirror_scale from tracker_service.py (tune distance in Python, not here)")]
    public bool usePythonMirrorHint = true;
    [Range(0f, 1f)]
    [Tooltip("Blend Python scale toward 1 (reduces double-distance amplification with normalized pose)")]
    public float pythonMirrorScaleInfluence = 0.55f;
    [Tooltip("Apply mirror_z_pull from Python (often feels twitchy; local Z uses 0 when off)")]
    public bool applyPythonZPull = false;
    [Tooltip("Extra Unity smoothing when using Python hint (sec). Lower = tighter to Python)")]
    public float pythonHintSmoothTime = 0.1f;
    [Tooltip("Max change in target scale per second (both Python and local mirror sizing)")]
    public float maxScaleStepPerSecond = 2.5f;
    [Tooltip("Used only if Python mirror hint is off. Face bbox height at comfortable distance")]
    [Range(0.12f, 0.55f)]
    public float referenceFaceHeight = 0.32f;
    [Tooltip("Shoulder-to-shoulder span in normalized image space at the same comfortable distance")]
    [Range(0.12f, 0.45f)]
    public float referenceShoulderSpan = 0.28f;
    [Range(0f, 1f)]
    [Tooltip("0 = face height only, 1 = shoulder span only; blend tracks whole-body distance better")]
    public float bodyVsFaceBlend = 0.45f;
    [Tooltip("Extra multiplier on top of distance ratio")]
    public float mirrorDistanceGain = 1f;
    [Tooltip("How fast scale catches up (seconds, lower = snappier)")]
    public float scaleSmoothTime = 0.22f;
    [Tooltip("Clamp final uniform scale")]
    public float minMirrorScale = 0.45f;
    public float maxMirrorScale = 2.6f;
    [Tooltip("Move toward camera when you step closer (uses local Z; flip sign if it goes the wrong way)")]
    public float zPullWhenClose = 0.25f;
    [Tooltip("Base uniform scale when mirror sizing is off")]
    public float staticUniformScale = 1f;
    [Header("Smoothing")]
    [Tooltip("Joint position lag (seconds). Higher = smoother, more delay")]
    public float jointSmoothTime = 0.14f;
    [Tooltip("Max units/sec joints can move (reduces spikes)")]
    public float jointMaxSpeed = 6f;

    [Header("Lines (bones)")]
    public bool drawBones = true;
    public float boneWidth = 0.015f;
    public Material boneMaterial;
    public Color boneColor = new Color(0.2f, 1f, 0.4f, 1f);

    private Transform[] _joints;
    private Vector3[] _smoothed;
    private Vector3[] _jointVel;
    private Renderer[] _jointSphereRenderers;
    private LineRenderer[] _bones;
    private float _smoothedDistanceScale = 1f;
    private float _smoothedZOffset;
    private float _scaleVel;
    private float _zVel;
    private Vector3 _baseLocalPos;
    private TrackingReceiver.LandmarkData[] _poseLatch;
    private bool _poseLatchValid;
    private MirrorMotionShared.RuntimeState _motionState;
    public float DebugEffectiveScale => _smoothedDistanceScale;
    public float DebugEffectiveZ => _smoothedZOffset;
    private static readonly (int a, int b)[] BonePairs =
    {
        (0, 1), (0, 2), (1, 2),
        (1, 3), (3, 5), (2, 4), (4, 6),
        (1, 7), (2, 8), (7, 8),
        (7, 9), (9, 11), (8, 10), (10, 12),
    };

    private static readonly string[] JointNames =
    {
        "Nose", "L_Shoulder", "R_Shoulder", "L_Elbow", "R_Elbow", "L_Wrist", "R_Wrist",
        "L_Hip", "R_Hip", "L_Knee", "R_Knee", "L_Ankle", "R_Ankle",
    };

    private void Start()
    {
        _baseLocalPos = transform.localPosition;
        if (autoBuildOnStart)
            BuildRig();
    }

    [ContextMenu("Build / Rebuild stick rig")]
    public void BuildRig()
    {
        ClearChildren();

        _joints = new Transform[JointNames.Length];
        _smoothed = new Vector3[JointNames.Length];
        _jointVel = new Vector3[JointNames.Length];
        _jointSphereRenderers = new Renderer[JointNames.Length];

        for (int i = 0; i < JointNames.Length; i++)
        {
            var go = new GameObject(JointNames[i]);
            go.transform.SetParent(transform, false);
            _joints[i] = go.transform;
            _smoothed[i] = Vector3.zero;

            if (addJointSpheres)
            {
                var sphere = GameObject.CreatePrimitive(PrimitiveType.Sphere);
                sphere.name = "Visual";
                sphere.transform.SetParent(go.transform, false);
                sphere.transform.localScale = Vector3.one * jointSphereScale;
                var col = sphere.GetComponent<Collider>();
                if (col != null) Destroy(col);

                var r = sphere.GetComponent<Renderer>();
                _jointSphereRenderers[i] = r;
                if (r != null && jointMaterial == null)
                {
                    Shader sh = Shader.Find("Universal Render Pipeline/Lit")
                              ?? Shader.Find("HDRP/Lit")
                              ?? Shader.Find("Standard");
                    if (sh == null) sh = Shader.Find("Unlit/Color");
                    var mat = new Material(sh);
                    if (mat.HasProperty("_BaseColor")) mat.SetColor("_BaseColor", new Color(0.9f, 0.9f, 0.95f));
                    else if (mat.HasProperty("_Color")) mat.color = new Color(0.9f, 0.9f, 0.95f);
                    r.sharedMaterial = mat;
                }
                else if (r != null) r.sharedMaterial = jointMaterial;
            }
        }

        if (drawBones)
            CreateBoneLines();
    }

    private void ClearChildren()
    {
        for (int i = transform.childCount - 1; i >= 0; i--)
        {
            var c = transform.GetChild(i);
            if (Application.isPlaying) Destroy(c.gameObject);
            else DestroyImmediate(c.gameObject);
        }

        if (_bones != null)
        {
            foreach (var lr in _bones)
            {
                if (lr != null && Application.isPlaying) Destroy(lr.gameObject);
                else if (lr != null) DestroyImmediate(lr.gameObject);
            }
            _bones = null;
        }
    }

    private void CreateBoneLines()
    {
        _bones = new LineRenderer[BonePairs.Length];
        var boneRoot = new GameObject("Bones");
        boneRoot.transform.SetParent(transform, false);

        Shader shader = Shader.Find("Sprites/Default");
        if (shader == null) shader = Shader.Find("Unlit/Color");

        for (int i = 0; i < BonePairs.Length; i++)
        {
            var go = new GameObject($"Bone_{BonePairs[i].a}_{BonePairs[i].b}");
            go.transform.SetParent(boneRoot.transform, false);
            var lr = go.AddComponent<LineRenderer>();
            lr.positionCount = 2;
            lr.startWidth = boneWidth;
            lr.endWidth = boneWidth;
            lr.useWorldSpace = true;
            lr.material = boneMaterial != null
                ? boneMaterial
                : new Material(shader) { color = boneColor };
            lr.startColor = boneColor;
            lr.endColor = boneColor;
            _bones[i] = lr;
        }
    }

    private static float PoseHeightProxy(List<TrackingReceiver.LandmarkData> pose)
    {
        if (pose == null || pose.Count < 13) return 0.2f;
        float noseY = pose[0].y;
        float anklesY = (pose[11].y + pose[12].y) * 0.5f;
        return Mathf.Abs(noseY - anklesY);
    }

    private static float PoseHeightProxyLatch(TrackingReceiver.LandmarkData[] pose)
    {
        if (pose == null || pose.Length < 13) return 0.2f;
        float noseY = pose[0].y;
        float anklesY = (pose[11].y + pose[12].y) * 0.5f;
        return Mathf.Abs(noseY - anklesY);
    }

    private static float ShoulderSpanNorm(List<TrackingReceiver.LandmarkData> pose, bool mirrorX)
    {
        if (pose == null || pose.Count < 3) return 0.2f;
        float x1 = pose[1].x;
        float x2 = pose[2].x;
        if (mirrorX)
        {
            x1 = 1f - x1;
            x2 = 1f - x2;
        }
        float dx = x2 - x1;
        float dy = pose[2].y - pose[1].y;
        return Mathf.Sqrt(dx * dx + dy * dy);
    }

    private static float ShoulderSpanNormLatch(TrackingReceiver.LandmarkData[] pose, bool mirrorX)
    {
        if (pose == null || pose.Length < 3) return 0.2f;
        float x1 = pose[1].x;
        float x2 = pose[2].x;
        if (mirrorX)
        {
            x1 = 1f - x1;
            x2 = 1f - x2;
        }
        float dx = x2 - x1;
        float dy = pose[2].y - pose[1].y;
        return Mathf.Sqrt(dx * dx + dy * dy);
    }

    private MirrorMotionShared.Settings BuildMotionSettings()
    {
        return new MirrorMotionShared.Settings
        {
            mirrorX = mirrorX,
            clampToVideoFrame = clampToVideoFrame,
            frameEdgeMargin = frameEdgeMargin,
            matchPreviewPixelAspect = matchPreviewPixelAspect,
            widthScale = widthScale,
            heightScale = heightScale,
            depthScale = depthScale,
            invertZ = false,
            mirrorDistanceSizing = mirrorDistanceSizing,
            usePythonMirrorHint = usePythonMirrorHint,
            pythonMirrorScaleInfluence = pythonMirrorScaleInfluence,
            applyPythonZPull = applyPythonZPull,
            pythonHintSmoothTime = pythonHintSmoothTime,
            maxScaleStepPerSecond = maxScaleStepPerSecond,
            referenceFaceHeight = referenceFaceHeight,
            referenceShoulderSpan = referenceShoulderSpan,
            bodyVsFaceBlend = bodyVsFaceBlend,
            mirrorDistanceGain = mirrorDistanceGain,
            scaleSmoothTime = scaleSmoothTime,
            minMirrorScale = minMirrorScale,
            maxMirrorScale = maxMirrorScale,
            zPullWhenClose = zPullWhenClose,
            staticUniformScale = staticUniformScale,
        };
    }

    private void EnsurePoseLatch()
    {
        if (_poseLatch != null) return;
        _poseLatch = new TrackingReceiver.LandmarkData[13];
        for (int i = 0; i < 13; i++)
            _poseLatch[i] = new TrackingReceiver.LandmarkData();
    }

    private void CopyLatchFrom(List<TrackingReceiver.LandmarkData> src)
    {
        for (int i = 0; i < 13; i++)
        {
            var s = src[i];
            _poseLatch[i].x = s.x;
            _poseLatch[i].y = s.y;
            _poseLatch[i].z = s.z;
            _poseLatch[i].visibility = s.visibility;
        }
        _poseLatchValid = true;
    }

    private static float EffectiveMinVis(float a, float b)
    {
        return Mathf.Min(Mathf.Clamp01(a), Mathf.Clamp01(b));
    }

    private static bool InVideoFrame(float nx, float ny, float margin)
    {
        return nx >= margin && nx <= 1f - margin && ny >= margin && ny <= 1f - margin;
    }

    private void LateUpdate()
    {
        if (receiver == null || _joints == null || _joints.Length == 0) return;

        var frame = receiver.LatestFrame;
        if (frame == null) return;

        EnsurePoseLatch();

        bool haveFreshPose = frame.poses != null && frame.poses.Count > 0 && frame.poses[0] != null
            && frame.poses[0].Count >= 13;
        if (haveFreshPose)
            CopyLatchFrom(frame.poses[0]);

        if (!_poseLatchValid) return;

        bool subjectVisible = !hideWhenSubjectOffscreen || IsSubjectVisible(frame, _poseLatch, BuildMotionSettings());
        if (!receiver.HasRecentData()) subjectVisible = false;
        SetStickVisible(subjectVisible);
        if (!subjectVisible) return;

        var motion = BuildMotionSettings();
        MirrorMotionShared.StepDistance(frame, _poseLatch, motion, Time.deltaTime, ref _motionState, out _smoothedDistanceScale, out _smoothedZOffset);
        transform.localScale = Vector3.one * _smoothedDistanceScale;
        var lp = _baseLocalPos;
        lp.z += _smoothedZOffset;
        transform.localPosition = lp;

        float m = frameEdgeMargin;
        float jt = Mathf.Max(0.02f, jointSmoothTime);

        for (int i = 0; i < 13; i++)
        {
            var lm = _poseLatch[i];
            var target = MirrorMotionShared.MapLandmarkToLocal(lm, frame, motion);

            _smoothed[i] = Vector3.SmoothDamp(
                _smoothed[i], target, ref _jointVel[i], jt, jointMaxSpeed, Time.deltaTime);
            _joints[i].localPosition = _smoothed[i];

            if (addJointSpheres && _jointSphereRenderers != null && i < _jointSphereRenderers.Length)
            {
                var r = _jointSphereRenderers[i];
                if (r != null)
                {
                    bool visOk = !hideLowVisibilityLimbs || _poseLatch[i].visibility >= minLimbVisibility;
                    if (!visOk)
                        r.enabled = false;
                    else if (clampToVideoFrame || !hideJointSpheresOutsideFrame)
                        r.enabled = true;
                    else
                    {
                        var p = _poseLatch[i];
                        float px = mirrorX ? 1f - p.x : p.x;
                        r.enabled = InVideoFrame(px, p.y, m);
                    }
                }
            }
        }

        if (drawBones && _bones != null)
        {
            bool InsideRaw(int idx)
            {
                var p = _poseLatch[idx];
                float px = mirrorX ? 1f - p.x : p.x;
                return InVideoFrame(px, p.y, m);
            }

            for (int b = 0; b < _bones.Length; b++)
            {
                var (a, c) = BonePairs[b];
                if (_bones[b] == null || a >= _joints.Length || c >= _joints.Length) continue;

                bool show = true;
                if (hideLowVisibilityLimbs
                    && EffectiveMinVis(_poseLatch[a].visibility, _poseLatch[c].visibility) < minLimbVisibility)
                {
                    show = false;
                }
                else if (clampToVideoFrame)
                {
                    // Joints are clamped to the frame; bones always form a connected skeleton
                    // (torso shortens along the edge instead of vanishing).
                    show = true;
                }
                else if (hideBonesWhenBothEndsOffScreen)
                {
                    show = InsideRaw(a) || InsideRaw(c);
                }

                _bones[b].enabled = show;
                if (!show) continue;

                Vector3 wa = _joints[a].position;
                Vector3 wc = _joints[c].position;
                _bones[b].SetPosition(0, wa);
                _bones[b].SetPosition(1, wc);
            }
        }
    }

    private bool IsSubjectVisible(
        TrackingReceiver.TrackingFrameData frame,
        TrackingReceiver.LandmarkData[] pose,
        MirrorMotionShared.Settings motion)
    {
        if (frame.faces != null && frame.faces.Count > 0)
        {
            var f = frame.faces[0];
            if (f.w > 0.02f && f.h > 0.02f)
            {
                float cx = motion.mirrorX ? 1f - f.cx : f.cx;
                bool bboxIn = cx >= motion.frameEdgeMargin
                    && cx <= 1f - motion.frameEdgeMargin
                    && f.cy >= motion.frameEdgeMargin
                    && f.cy <= 1f - motion.frameEdgeMargin;
                if (bboxIn) return true;
            }
        }

        int[] core = { 0, 1, 2, 7, 8 };
        for (int i = 0; i < core.Length; i++)
        {
            int idx = core[i];
            if (idx >= pose.Length) continue;
            if (pose[idx].visibility < minLimbVisibility) continue;
            if (MirrorMotionShared.IsInsideFrame(pose[idx], motion)) return true;
        }

        return false;
    }

    private void SetStickVisible(bool visible)
    {
        if (addJointSpheres && _jointSphereRenderers != null)
        {
            for (int i = 0; i < _jointSphereRenderers.Length; i++)
            {
                var r = _jointSphereRenderers[i];
                if (r != null) r.enabled = visible;
            }
        }

        if (_bones != null)
        {
            for (int i = 0; i < _bones.Length; i++)
            {
                if (_bones[i] != null) _bones[i].enabled = visible;
            }
        }
    }
}
