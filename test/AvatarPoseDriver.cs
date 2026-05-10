using System.Collections.Generic;
using UnityEngine;

/// <summary>
/// Drives a Humanoid avatar from TrackingReceiver compact pose (13 joints).
/// Attach to the avatar root that has an Animator.
/// </summary>
public class AvatarPoseDriver : MonoBehaviour
{
    public TrackingReceiver receiver;
    public Animator animator;

    [Header("Tracking Space")]
    public bool mirrorX = true;
    [Tooltip("Flip tracker depth direction if avatar appears inside-out on turns.")]
    public bool invertZ = true;
    public bool matchPreviewPixelAspect = true;
    public float widthScale = 1.6f;
    public float heightScale = 1.8f;
    public float depthScale = 0.8f;
    public bool clampToVideoFrame = true;
    [Range(0f, 0.15f)] public float frameEdgeMargin = 0.02f;
    [Tooltip("Hide limb rotations when either endpoint is outside camera frame.")]
    public bool hideOffscreenLimbs = true;
    [Tooltip("Hide avatar mesh when tracked subject leaves camera frame.")]
    public bool hideWhenSubjectOffscreen = true;

    [Header("Pose Filtering")]
    [Range(0f, 1f)] public float minVisibility = 0.3f;
    public float rotationLerpSpeed = 12f;
    [Tooltip("Hard limit from bind pose to prevent impossible contortion.")]
    [Range(20f, 140f)] public float maxBoneAngleFromBind = 85f;
    [Header("Head Tracking")]
    public bool enableHeadTracking = true;
    [Tooltip("How strongly nose left/right steers head yaw.")]
    public float headYawGain = 140f;
    [Tooltip("How strongly nose up/down steers head pitch.")]
    public float headPitchGain = 110f;
    [Tooltip("How strongly shoulder tilt steers head roll.")]
    public float headRollGain = 35f;
    public float headMaxYaw = 55f;
    public float headMaxPitch = 40f;
    public float headMaxRoll = 28f;
    public float headLerpSpeed = 14f;
    public bool followHipsPosition = false;
    public float hipsPositionLerpSpeed = 8f;

    [Header("Mirror Distance (like stick rig)")]
    public bool mirrorDistanceSizing = true;
    public bool usePythonMirrorHint = true;
    [Range(0f, 1f)] public float pythonMirrorScaleInfluence = 0.55f;
    public bool applyPythonZPull = false;
    public float pythonHintSmoothTime = 0.1f;
    public float maxScaleStepPerSecond = 2.5f;
    public float referenceFaceHeight = 0.32f;
    public float referenceShoulderSpan = 0.28f;
    [Range(0f, 1f)] public float bodyVsFaceBlend = 0.45f;
    public float mirrorDistanceGain = 1f;
    public float scaleSmoothTime = 0.22f;
    public float minMirrorScale = 0.45f;
    public float maxMirrorScale = 2.6f;
    public float zPullWhenClose = 0.25f;

    [Header("Root Motion (green rig style)")]
    public bool moveRootLikeStickRig = true;
    [Tooltip("If true, root snaps to target every frame (teleport style).")]
    public bool teleportRoot = false;
    public float rootLerpSpeed = 14f;
    [Tooltip("Snap when target jumps more than this local distance.")]
    public float rootTeleportDistance = 0.35f;
    [Tooltip("How much landmark Z contributes to root movement.")]
    [Range(0f, 1f)] public float rootDepthWeight = 0.35f;

    private readonly TrackingReceiver.LandmarkData[] _poseLatch = new TrackingReceiver.LandmarkData[13];
    private bool _poseValid;

    private Transform _hips;
    private Transform _spine;
    private Transform _chest;
    private Transform _head;
    private Transform _lUpperArm;
    private Transform _lLowerArm;
    private Transform _rUpperArm;
    private Transform _rLowerArm;
    private Transform _lUpperLeg;
    private Transform _lLowerLeg;
    private Transform _rUpperLeg;
    private Transform _rLowerLeg;

    private readonly Dictionary<Transform, Quaternion> _bindLocalRot = new Dictionary<Transform, Quaternion>();
    private readonly Dictionary<Transform, Vector3> _bindAimLocal = new Dictionary<Transform, Vector3>();
    private float _smoothedDistanceScale = 1f;
    private float _smoothedZOffset;
    private float _scaleVel;
    private float _zVel;
    private Vector3 _baseLocalPos;
    private Vector3 _smoothedRootOffset;
    private MirrorMotionShared.RuntimeState _motionState;
    private Renderer[] _avatarRenderers;
    public float DebugEffectiveScale => _smoothedDistanceScale;
    public float DebugEffectiveZ => _smoothedZOffset;
    public Vector3 DebugRootOffset => _smoothedRootOffset;

    private static readonly int Nose = 0;
    private static readonly int LShoulder = 1;
    private static readonly int RShoulder = 2;
    private static readonly int LElbow = 3;
    private static readonly int RElbow = 4;
    private static readonly int LWrist = 5;
    private static readonly int RWrist = 6;
    private static readonly int LHip = 7;
    private static readonly int RHip = 8;
    private static readonly int LKnee = 9;
    private static readonly int RKnee = 10;
    private static readonly int LAnkle = 11;
    private static readonly int RAnkle = 12;

    private void Awake()
    {
        if (animator == null) animator = GetComponent<Animator>();
        if (animator == null)
        {
            Debug.LogError("[AvatarPoseDriver] Animator not found.");
            enabled = false;
            return;
        }

        for (int i = 0; i < _poseLatch.Length; i++)
            _poseLatch[i] = new TrackingReceiver.LandmarkData();
        _baseLocalPos = transform.localPosition;
        _avatarRenderers = GetComponentsInChildren<Renderer>(true);

        CacheHumanoidBones();
        CaptureBindData(_lUpperArm, _lLowerArm);
        CaptureBindData(_lLowerArm, animator.GetBoneTransform(HumanBodyBones.LeftHand));
        CaptureBindData(_rUpperArm, _rLowerArm);
        CaptureBindData(_rLowerArm, animator.GetBoneTransform(HumanBodyBones.RightHand));
        CaptureBindData(_lUpperLeg, _lLowerLeg);
        CaptureBindData(_lLowerLeg, animator.GetBoneTransform(HumanBodyBones.LeftFoot));
        CaptureBindData(_rUpperLeg, _rLowerLeg);
        CaptureBindData(_rLowerLeg, animator.GetBoneTransform(HumanBodyBones.RightFoot));
        CaptureBindData(_spine, _chest);
        CaptureBindData(_chest, _head);
        CaptureBindData(_head, null);
    }

    private void CacheHumanoidBones()
    {
        _hips = animator.GetBoneTransform(HumanBodyBones.Hips);
        _spine = animator.GetBoneTransform(HumanBodyBones.Spine);
        _chest = animator.GetBoneTransform(HumanBodyBones.Chest);
        _head = animator.GetBoneTransform(HumanBodyBones.Head);
        _lUpperArm = animator.GetBoneTransform(HumanBodyBones.LeftUpperArm);
        _lLowerArm = animator.GetBoneTransform(HumanBodyBones.LeftLowerArm);
        _rUpperArm = animator.GetBoneTransform(HumanBodyBones.RightUpperArm);
        _rLowerArm = animator.GetBoneTransform(HumanBodyBones.RightLowerArm);
        _lUpperLeg = animator.GetBoneTransform(HumanBodyBones.LeftUpperLeg);
        _lLowerLeg = animator.GetBoneTransform(HumanBodyBones.LeftLowerLeg);
        _rUpperLeg = animator.GetBoneTransform(HumanBodyBones.RightUpperLeg);
        _rLowerLeg = animator.GetBoneTransform(HumanBodyBones.RightLowerLeg);
    }

    private void LateUpdate()
    {
        if (receiver == null || animator == null) return;
        var frame = receiver.LatestFrame;
        if (frame == null) return;

        bool fresh = frame.poses != null && frame.poses.Count > 0 && frame.poses[0] != null && frame.poses[0].Count >= 13;
        if (fresh)
        {
            for (int i = 0; i < 13; i++)
            {
                var s = frame.poses[0][i];
                _poseLatch[i].x = s.x;
                _poseLatch[i].y = s.y;
                _poseLatch[i].z = s.z;
                _poseLatch[i].visibility = s.visibility;
            }
            _poseValid = true;
        }
        if (!_poseValid) return;

        bool subjectVisible = !hideWhenSubjectOffscreen || IsSubjectVisible(frame);
        if (!receiver.HasRecentData()) subjectVisible = false;
        SetAvatarVisible(subjectVisible);
        if (!subjectVisible) return;

        ApplyMirrorDistance(frame);
        ApplyRootMotion(frame);
        Vector3[] p = BuildTrackedPoints(frame);
        // Torso
        Vector3 shoulderCenter = (p[(int)LShoulder] + p[(int)RShoulder]) * 0.5f;
        Vector3 hipCenter = (p[(int)LHip] + p[(int)RHip]) * 0.5f;
        RotateBoneToDirection(_spine, shoulderCenter - hipCenter);
        RotateBoneToDirection(_chest, shoulderCenter - hipCenter);

        // Head looks from shoulder-center toward nose.
        if (enableHeadTracking && _head != null && IsVisible(Nose) && IsVisible(LShoulder) && IsVisible(RShoulder))
        {
            RotateHeadFromPose();
        }

        // Arms
        SolveLimb(_lUpperArm, _lLowerArm, p[(int)LShoulder], p[(int)LElbow], p[(int)LWrist], LShoulder, LElbow, LWrist);
        SolveLimb(_rUpperArm, _rLowerArm, p[(int)RShoulder], p[(int)RElbow], p[(int)RWrist], RShoulder, RElbow, RWrist);

        // Legs
        SolveLimb(_lUpperLeg, _lLowerLeg, p[(int)LHip], p[(int)LKnee], p[(int)LAnkle], LHip, LKnee, LAnkle);
        SolveLimb(_rUpperLeg, _rLowerLeg, p[(int)RHip], p[(int)RKnee], p[(int)RAnkle], RHip, RKnee, RAnkle);

        if (followHipsPosition && _hips != null && IsVisible(LHip) && IsVisible(RHip))
        {
            Vector3 hipsTarget = (p[(int)LHip] + p[(int)RHip]) * 0.5f;
            float t = 1f - Mathf.Exp(-hipsPositionLerpSpeed * Time.deltaTime);
            _hips.position = Vector3.Lerp(_hips.position, hipsTarget, t);
        }
    }

    private Vector3[] BuildTrackedPoints(TrackingReceiver.TrackingFrameData frame)
    {
        Vector3[] outPts = new Vector3[13];
        var motion = BuildMotionSettings();

        for (int i = 0; i < 13; i++)
        {
            Vector3 local = MirrorMotionShared.MapLandmarkToLocal(_poseLatch[i], frame, motion);
            outPts[i] = transform.TransformPoint(local);
        }

        return outPts;
    }

    private void SolveLimb(
        Transform upper,
        Transform lower,
        Vector3 a,
        Vector3 b,
        Vector3 c,
        int ia,
        int ib,
        int ic)
    {
        if (upper != null && IsVisible(ia) && IsVisible(ib) && IsInFrame(ia) && IsInFrame(ib))
            RotateBoneToDirection(upper, b - a);
        if (lower != null && IsVisible(ib) && IsVisible(ic) && IsInFrame(ib) && IsInFrame(ic))
            RotateBoneToDirection(lower, c - b);
    }

    private bool IsVisible(int idx)
    {
        return _poseLatch[idx].visibility >= minVisibility;
    }

    private bool IsInFrame(int idx)
    {
        if (!hideOffscreenLimbs) return true;
        return MirrorMotionShared.IsInsideFrame(_poseLatch[idx], BuildMotionSettings());
    }

    private bool IsSubjectVisible(TrackingReceiver.TrackingFrameData frame)
    {
        var motion = BuildMotionSettings();

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

        int[] core = { Nose, LShoulder, RShoulder, LHip, RHip };
        for (int i = 0; i < core.Length; i++)
        {
            int idx = core[i];
            if (_poseLatch[idx].visibility < minVisibility) continue;
            if (MirrorMotionShared.IsInsideFrame(_poseLatch[idx], motion)) return true;
        }

        return false;
    }

    private void SetAvatarVisible(bool visible)
    {
        if (_avatarRenderers == null) return;
        for (int i = 0; i < _avatarRenderers.Length; i++)
        {
            var r = _avatarRenderers[i];
            if (r != null) r.enabled = visible;
        }
    }

    private static float PoseHeightProxy(TrackingReceiver.LandmarkData[] pose)
    {
        float noseY = pose[Nose].y;
        float anklesY = (pose[LAnkle].y + pose[RAnkle].y) * 0.5f;
        return Mathf.Abs(noseY - anklesY);
    }

    private static float ShoulderSpanNorm(TrackingReceiver.LandmarkData[] pose, bool mirrorX)
    {
        float x1 = pose[LShoulder].x;
        float x2 = pose[RShoulder].x;
        if (mirrorX)
        {
            x1 = 1f - x1;
            x2 = 1f - x2;
        }
        float dx = x2 - x1;
        float dy = pose[RShoulder].y - pose[LShoulder].y;
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
            invertZ = invertZ,
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
            staticUniformScale = 1f,
        };
    }

    private void ApplyMirrorDistance(TrackingReceiver.TrackingFrameData frame)
    {
        MirrorMotionShared.StepDistance(frame, _poseLatch, BuildMotionSettings(), Time.deltaTime, ref _motionState, out _smoothedDistanceScale, out _smoothedZOffset);
        transform.localScale = Vector3.one * _smoothedDistanceScale;
    }

    private void ApplyRootMotion(TrackingReceiver.TrackingFrameData frame)
    {
        Vector3 targetOffset = Vector3.zero;
        if (moveRootLikeStickRig)
        {
            float x = (_poseLatch[LHip].x + _poseLatch[RHip].x) * 0.5f;
            float y = (_poseLatch[LHip].y + _poseLatch[RHip].y) * 0.5f;
            float z = (_poseLatch[LHip].z + _poseLatch[RHip].z) * 0.5f;

            if (mirrorX) x = 1f - x;
            if (clampToVideoFrame)
            {
                float m = frameEdgeMargin;
                x = Mathf.Clamp(x, m, 1f - m);
                y = Mathf.Clamp(y, m, 1f - m);
            }

            MirrorMotionShared.ComputeEffectiveXYScale(frame, BuildMotionSettings(), out var effX, out var effY);

            float dz = invertZ ? -z : z;
            targetOffset = new Vector3(
                (x - 0.5f) * effX,
                (0.5f - y) * effY,
                dz * depthScale * Mathf.Clamp01(rootDepthWeight)
            );
        }

        if (teleportRoot || (_smoothedRootOffset - targetOffset).magnitude > Mathf.Max(0.01f, rootTeleportDistance))
        {
            _smoothedRootOffset = targetOffset;
        }
        else
        {
            float t = 1f - Mathf.Exp(-Mathf.Max(0.01f, rootLerpSpeed) * Time.deltaTime);
            _smoothedRootOffset = Vector3.Lerp(_smoothedRootOffset, targetOffset, t);
        }

        var lp = _baseLocalPos + _smoothedRootOffset;
        lp.z += _smoothedZOffset;
        transform.localPosition = lp;
    }

    private void CaptureBindData(Transform bone, Transform child)
    {
        if (bone == null) return;
        _bindLocalRot[bone] = bone.localRotation;

        Vector3 dirLocal = Vector3.forward;
        if (child != null)
        {
            Vector3 dirWorld = child.position - bone.position;
            if (dirWorld.sqrMagnitude > 0.00001f)
                dirLocal = bone.InverseTransformDirection(dirWorld.normalized);
        }
        _bindAimLocal[bone] = dirLocal.normalized;
    }

    private void RotateHeadFromPose()
    {
        if (_head == null) return;
        if (!_bindLocalRot.TryGetValue(_head, out var bindLocal)) return;

        float sx = (_poseLatch[LShoulder].x + _poseLatch[RShoulder].x) * 0.5f;
        float sy = (_poseLatch[LShoulder].y + _poseLatch[RShoulder].y) * 0.5f;
        float noseX = _poseLatch[Nose].x;
        float noseY = _poseLatch[Nose].y;

        // Nose relative to shoulder center in normalized image space.
        float dx = noseX - sx;
        float dy = sy - noseY;
        if (mirrorX) dx = -dx;

        float yaw = Mathf.Clamp(dx * headYawGain, -headMaxYaw, headMaxYaw);
        float pitch = Mathf.Clamp(dy * headPitchGain, -headMaxPitch, headMaxPitch);

        // Shoulder slope gives a stable roll cue.
        float lsx = _poseLatch[LShoulder].x;
        float lsy = _poseLatch[LShoulder].y;
        float rsx = _poseLatch[RShoulder].x;
        float rsy = _poseLatch[RShoulder].y;
        float shoulderAngle = Mathf.Atan2(rsy - lsy, rsx - lsx) * Mathf.Rad2Deg;
        float roll = Mathf.Clamp(-shoulderAngle * (headRollGain / 45f), -headMaxRoll, headMaxRoll);

        Quaternion targetLocal = bindLocal * Quaternion.Euler(-pitch, yaw, roll);
        float t = 1f - Mathf.Exp(-Mathf.Max(0.01f, headLerpSpeed) * Time.deltaTime);
        _head.localRotation = Quaternion.Slerp(_head.localRotation, targetLocal, t);
    }

    private void RotateBoneToDirection(Transform bone, Vector3 dirWorld)
    {
        if (bone == null || bone.parent == null || dirWorld.sqrMagnitude < 0.00001f) return;
        if (!_bindLocalRot.TryGetValue(bone, out var bindLocal)) return;
        if (!_bindAimLocal.TryGetValue(bone, out var bindAim)) return;

        Vector3 dirLocalParent = bone.parent.InverseTransformDirection(dirWorld.normalized);
        if (dirLocalParent.sqrMagnitude < 0.00001f) return;

        Quaternion delta = Quaternion.FromToRotation(bindAim, dirLocalParent.normalized);
        Quaternion unclamped = delta * bindLocal;
        float angle = Quaternion.Angle(bindLocal, unclamped);
        Quaternion targetLocal = angle > maxBoneAngleFromBind
            ? Quaternion.Slerp(bindLocal, unclamped, maxBoneAngleFromBind / Mathf.Max(0.0001f, angle))
            : unclamped;

        float t = 1f - Mathf.Exp(-rotationLerpSpeed * Time.deltaTime);
        bone.localRotation = Quaternion.Slerp(bone.localRotation, targetLocal, t);
    }
}
