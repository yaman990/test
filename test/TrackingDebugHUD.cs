using UnityEngine;

public class TrackingDebugHUD : MonoBehaviour
{
    public TrackingReceiver receiver;
    public StickFigureDriver stickFigure;
    public AvatarPoseDriver avatar;
    public Vector2 offset = new Vector2(12f, 12f);
    public int fontSize = 18;
    public Color textColor = Color.white;

    private GUIStyle _style;

    private void Awake()
    {
        _style = new GUIStyle
        {
            fontSize = fontSize,
            normal = { textColor = textColor }
        };
    }

    private void OnGUI()
    {
        if (_style == null)
        {
            _style = new GUIStyle();
        }
        _style.fontSize = fontSize;
        _style.normal.textColor = textColor;

        float x = offset.x;
        float y = offset.y;
        float lineH = fontSize + 6f;

        if (receiver == null)
        {
            GUI.Label(new Rect(x, y, 700, lineH), "TrackingDebugHUD: assign a TrackingReceiver in Inspector.", _style);
            return;
        }

        bool alive = receiver.HasRecentData();
        var frame = receiver.LatestFrame;

        GUI.Label(new Rect(x, y, 700, lineH), $"Tracker: {(alive ? "LIVE" : "STALE/NO DATA")}", _style);
        y += lineH;

        if (frame == null)
        {
            GUI.Label(new Rect(x, y, 700, lineH), "No frame received yet.", _style);
            return;
        }

        GUI.Label(new Rect(x, y, 700, lineH), $"Seq: {frame.seq} | Timestamp: {frame.timestampMs}", _style);
        y += lineH;
        GUI.Label(new Rect(x, y, 700, lineH), $"Camera: {frame.cameraW}x{frame.cameraH}", _style);
        y += lineH;
        if (frame.hasMirrorHint)
            GUI.Label(new Rect(x, y, 700, lineH), $"Python mirror: scale={frame.mirrorScale:F2} z={frame.mirrorZPull:F2}", _style);
        else
            GUI.Label(new Rect(x, y, 700, lineH), "Python mirror: (not in packet — update tracker_service)", _style);
        y += lineH;
        GUI.Label(new Rect(x, y, 700, lineH), $"Faces: {frame.faces.Count} | Poses: {frame.poses.Count}", _style);
        y += lineH;

        if (stickFigure != null)
        {
            GUI.Label(
                new Rect(x, y, 900, lineH),
                $"Stick output: scale={stickFigure.DebugEffectiveScale:F3} z={stickFigure.DebugEffectiveZ:F3}",
                _style);
            y += lineH;
        }

        if (avatar != null)
        {
            var ro = avatar.DebugRootOffset;
            GUI.Label(
                new Rect(x, y, 1100, lineH),
                $"Avatar output: scale={avatar.DebugEffectiveScale:F3} z={avatar.DebugEffectiveZ:F3} root=({ro.x:F2},{ro.y:F2},{ro.z:F2})",
                _style);
            y += lineH;
        }

        if (frame.faces.Count > 0)
        {
            var f = frame.faces[0];
            GUI.Label(new Rect(x, y, 900, lineH), $"Face0 bbox: cx={f.cx:F3}, cy={f.cy:F3}, w={f.w:F3}, h={f.h:F3} | lm={f.landmarks.Count}", _style);
            y += lineH;
        }

        if (frame.poses.Count > 0)
        {
            GUI.Label(new Rect(x, y, 700, lineH), $"Pose0 joints: {frame.poses[0].Count}", _style);
        }
    }
}
