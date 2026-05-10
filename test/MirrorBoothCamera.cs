using UnityEngine;

/// <summary>
/// One-click-ish defaults for a "mirror booth" feel — no real-camera matching.
/// Put on Main Camera. Tweak FOV and height until it feels right for your TV/monitor.
/// </summary>
[ExecuteAlways]
public class MirrorBoothCamera : MonoBehaviour
{
    [Tooltip("Slightly wide = more of the body visible, arcade mirror vibe")]
    [Range(40f, 85f)]
    public float verticalFov = 58f;

    [Tooltip("Lens height in meters (eye-ish for standing adults)")]
    public float eyeHeightMeters = 1.55f;

    [Tooltip("How far in front of the lens the \"you\" spot is (meters). Bigger = figure smaller if you don't change scale.")]
    public float standDistanceMeters = 2.2f;

    [Tooltip("Look slightly down at where people stand")]
    [Range(0f, 20f)]
    public float pitchDownDegrees = 8f;

    [Tooltip("Near clip — keep small so close guests aren't cut off")]
    public float nearClip = 0.08f;

    public float farClip = 100f;

    [Tooltip("Apply in Edit mode too (for framing)")]
    public bool applyInEditMode = true;

    private void OnEnable()
    {
        Apply();
    }

    private void OnValidate()
    {
        if (applyInEditMode) Apply();
    }

    [ContextMenu("Apply mirror booth framing")]
    public void Apply()
    {
        var cam = GetComponent<Camera>();
        if (cam == null) return;

        cam.fieldOfView = verticalFov;
        cam.nearClipPlane = nearClip;
        cam.farClipPlane = farClip;

        // Camera back from the spot where the player stands, looking at that spot
        transform.position = new Vector3(0f, eyeHeightMeters, -standDistanceMeters);
        transform.rotation = Quaternion.Euler(pitchDownDegrees, 0f, 0f);
    }
}
