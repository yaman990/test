using UnityEngine;

/// <summary>
/// Sets Camera.fieldOfView (vertical, in degrees) from a real lens focal length and sensor size.
/// Use this so the Unity view matches your physical webcam / mirror booth camera.
/// </summary>
[ExecuteAlways]
[RequireComponent(typeof(Camera))]
public class PhysicalCameraMatch : MonoBehaviour
{
    public enum FovMode
    {
        FromSensorAndFocalLength,
        ManualVerticalDegrees
    }

    public FovMode mode = FovMode.FromSensorAndFocalLength;

    [Tooltip("Lens focal length in mm (from camera spec sheet)")]
    public float focalLengthMm = 2.8f;

    [Tooltip("Sensor width in mm (e.g. 1/3\" ~ 4.8mm, check datasheet)")]
    public float sensorWidthMm = 4.8f;

    [Tooltip("Sensor height in mm")]
    public float sensorHeightMm = 3.6f;

    [Tooltip("Used when mode is ManualVerticalDegrees")]
    [Range(10f, 120f)]
    public float verticalFovDegrees = 60f;

    [Tooltip("Recompute when values change (Editor + Play)")]
    public bool autoApply = true;

    private Camera _cam;

    private void OnEnable()
    {
        _cam = GetComponent<Camera>();
        Apply();
    }

    private void OnValidate()
    {
        if (autoApply && _cam == null) _cam = GetComponent<Camera>();
        if (autoApply) Apply();
    }

    [ContextMenu("Apply FOV from settings")]
    public void Apply()
    {
        if (_cam == null) _cam = GetComponent<Camera>();
        if (_cam == null) return;

        if (mode == FovMode.ManualVerticalDegrees)
        {
            _cam.fieldOfView = verticalFovDegrees;
            return;
        }

        focalLengthMm = Mathf.Max(0.01f, focalLengthMm);
        sensorHeightMm = Mathf.Max(0.01f, sensorHeightMm);

        // vertical FOV = 2 * atan( sensorHeight / (2 * f) )
        float halfHeight = sensorHeightMm * 0.5f;
        float verticalRad = 2f * Mathf.Atan(halfHeight / focalLengthMm);
        _cam.fieldOfView = verticalRad * Mathf.Rad2Deg;
    }

    /// <summary>
    /// Empirical FOV: place an object of realWidthMeters at distanceMeters, measure how many
    /// normalized units (0-1) it spans horizontally in your tracker image (or pixels / frame width).
    /// Call from a one-off calibration script if you don't trust datasheet numbers.
    /// </summary>
    public static float EstimateVerticalFovFromHorizontalSpan(
        float distanceMeters,
        float realWidthMeters,
        float normalizedWidth01,
        float aspectWidthOverHeight)
    {
        if (normalizedWidth01 <= 1e-4f || distanceMeters <= 1e-4f) return 60f;
        float halfHorizontalRad = Mathf.Atan((realWidthMeters * 0.5f) / distanceMeters);
        float fullHorizontalRad = 2f * halfHorizontalRad;
        float fullVerticalRad = fullHorizontalRad / aspectWidthOverHeight;
        return fullVerticalRad * Mathf.Rad2Deg;
    }
}
