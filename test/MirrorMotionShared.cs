using UnityEngine;

public static class MirrorMotionShared
{
    public struct Settings
    {
        public bool mirrorX;
        public bool clampToVideoFrame;
        public float frameEdgeMargin;
        public bool matchPreviewPixelAspect;
        public float widthScale;
        public float heightScale;
        public float depthScale;
        public bool invertZ;

        public bool mirrorDistanceSizing;
        public bool usePythonMirrorHint;
        public float pythonMirrorScaleInfluence;
        public bool applyPythonZPull;
        public float pythonHintSmoothTime;
        public float maxScaleStepPerSecond;
        public float referenceFaceHeight;
        public float referenceShoulderSpan;
        public float bodyVsFaceBlend;
        public float mirrorDistanceGain;
        public float scaleSmoothTime;
        public float minMirrorScale;
        public float maxMirrorScale;
        public float zPullWhenClose;
        public float staticUniformScale;
    }

    public struct RuntimeState
    {
        public float smoothedDistanceScale;
        public float smoothedZOffset;
        public float scaleVel;
        public float zVel;
        public bool initialized;
    }

    public static void EnsureInitialized(ref RuntimeState st)
    {
        if (st.initialized) return;
        st.smoothedDistanceScale = 1f;
        st.smoothedZOffset = 0f;
        st.scaleVel = 0f;
        st.zVel = 0f;
        st.initialized = true;
    }

    public static Vector3 MapLandmarkToLocal(
        TrackingReceiver.LandmarkData lm,
        TrackingReceiver.TrackingFrameData frame,
        in Settings s)
    {
        float x = lm.x;
        float y = lm.y;
        float z = lm.z;
        if (s.mirrorX) x = 1f - x;
        if (s.clampToVideoFrame)
        {
            float m = s.frameEdgeMargin;
            x = Mathf.Clamp(x, m, 1f - m);
            y = Mathf.Clamp(y, m, 1f - m);
        }

        ComputeEffectiveXYScale(frame, s, out var effX, out var effY);
        float dz = s.invertZ ? -z : z;
        return new Vector3(
            (x - 0.5f) * effX,
            (0.5f - y) * effY,
            dz * s.depthScale
        );
    }

    public static bool IsInsideFrame(TrackingReceiver.LandmarkData lm, in Settings s)
    {
        float x = lm.x;
        float y = lm.y;
        if (s.mirrorX) x = 1f - x;
        float m = s.frameEdgeMargin;
        return x >= m && x <= 1f - m && y >= m && y <= 1f - m;
    }

    public static void ComputeEffectiveXYScale(
        TrackingReceiver.TrackingFrameData frame,
        in Settings s,
        out float effX,
        out float effY)
    {
        if (s.matchPreviewPixelAspect && frame.cameraW > 0 && frame.cameraH > 0)
        {
            float aspect = (float)frame.cameraW / Mathf.Max(1, frame.cameraH);
            effY = s.heightScale;
            effX = s.heightScale * aspect;
        }
        else
        {
            effX = s.widthScale;
            effY = s.heightScale;
        }
    }

    public static void StepDistance(
        TrackingReceiver.TrackingFrameData frame,
        TrackingReceiver.LandmarkData[] poseLatch,
        in Settings s,
        float deltaTime,
        ref RuntimeState st,
        out float outScale,
        out float outZ)
    {
        EnsureInitialized(ref st);

        if (!s.mirrorDistanceSizing)
        {
            st.smoothedDistanceScale = s.staticUniformScale;
            st.smoothedZOffset = 0f;
            outScale = st.smoothedDistanceScale;
            outZ = st.smoothedZOffset;
            return;
        }

        float targetScale;
        float zTarget;
        float stTime;
        if (s.usePythonMirrorHint && frame.hasMirrorHint)
        {
            float raw = frame.mirrorScale;
            if (!float.IsFinite(raw)) raw = st.smoothedDistanceScale;
            raw = Mathf.Clamp(raw, s.minMirrorScale, s.maxMirrorScale);
            float inf = Mathf.Clamp01(s.pythonMirrorScaleInfluence);
            targetScale = Mathf.Lerp(1f, raw, inf);
            float mz = frame.mirrorZPull;
            if (!float.IsFinite(mz)) mz = 0f;
            zTarget = s.applyPythonZPull ? mz : 0f;
            stTime = Mathf.Max(0.02f, s.pythonHintSmoothTime);
        }
        else
        {
            float faceH = frame.faces.Count > 0 && frame.faces[0].h > 0.05f
                ? frame.faces[0].h
                : PoseHeightProxy(poseLatch);
            float shoulderW = ShoulderSpanNorm(poseLatch, s.mirrorX);
            float bodyH = PoseHeightProxy(poseLatch);

            faceH = Mathf.Max(faceH, 0.06f);
            shoulderW = Mathf.Max(shoulderW, 0.06f);
            bodyH = Mathf.Max(bodyH, 0.1f);

            float ratioFace = faceH / Mathf.Max(s.referenceFaceHeight, 0.08f);
            float ratioShoulder = shoulderW / Mathf.Max(s.referenceShoulderSpan, 0.08f);
            float ratioBody = bodyH / Mathf.Max(s.referenceFaceHeight * 2.2f, 0.15f);
            float ratioBlend = Mathf.Lerp(ratioFace, ratioShoulder, s.bodyVsFaceBlend);
            ratioBlend = Mathf.Lerp(ratioBlend, ratioBody, 0.25f);

            targetScale = Mathf.Clamp(ratioBlend * s.mirrorDistanceGain, s.minMirrorScale, s.maxMirrorScale);
            zTarget = -(ratioBlend - 1f) * s.zPullWhenClose;
            stTime = Mathf.Max(0.02f, s.scaleSmoothTime);
        }

        float maxStep = Mathf.Max(0.01f, s.maxScaleStepPerSecond) * deltaTime;
        targetScale = Mathf.Clamp(targetScale, st.smoothedDistanceScale - maxStep, st.smoothedDistanceScale + maxStep);

        st.smoothedDistanceScale = Mathf.SmoothDamp(
            st.smoothedDistanceScale, targetScale, ref st.scaleVel, stTime, Mathf.Infinity, deltaTime);
        st.smoothedZOffset = Mathf.SmoothDamp(
            st.smoothedZOffset, zTarget, ref st.zVel, stTime, Mathf.Infinity, deltaTime);

        outScale = st.smoothedDistanceScale;
        outZ = st.smoothedZOffset;
    }

    private static float PoseHeightProxy(TrackingReceiver.LandmarkData[] pose)
    {
        if (pose == null || pose.Length < 13) return 0.2f;
        float noseY = pose[0].y;
        float anklesY = (pose[11].y + pose[12].y) * 0.5f;
        return Mathf.Abs(noseY - anklesY);
    }

    private static float ShoulderSpanNorm(TrackingReceiver.LandmarkData[] pose, bool mirrorX)
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
}
