// Read capture capability tables without changing device controls.
using System;
using System.Collections.Generic;
using System.Runtime.InteropServices;
using System.Runtime.InteropServices.ComTypes;

namespace CameraDiagnostics {
    [ComImport, Guid("56A86895-0AD4-11CE-B03A-0020AF0BA770"), InterfaceType(ComInterfaceType.InterfaceIsIUnknown)]
    interface IBaseFilter { }
    [ComImport, Guid("56A868A9-0AD4-11CE-B03A-0020AF0BA770"), InterfaceType(ComInterfaceType.InterfaceIsIUnknown)]
    interface IGraphBuilder { }
    [ComImport, Guid("29840822-5B84-11D0-BD3B-00A0C911CE86"), InterfaceType(ComInterfaceType.InterfaceIsIUnknown)]
    interface ICreateDevEnum {
        [PreserveSig] int CreateClassEnumerator([In] ref Guid category, out IEnumMoniker enumerator, int flags);
    }
    [ComImport, Guid("93E5A4E0-2D50-11D2-ABFA-00A0C9C6E38D"), InterfaceType(ComInterfaceType.InterfaceIsIUnknown)]
    interface ICaptureGraphBuilder2 {
        [PreserveSig] int SetFiltergraph(IGraphBuilder graph);
        [PreserveSig] int GetFiltergraph([MarshalAs(UnmanagedType.Interface)] out object graph);
        [PreserveSig] int SetOutputFileName([In] ref Guid type, [MarshalAs(UnmanagedType.LPWStr)] string name,
                                           [MarshalAs(UnmanagedType.Interface)] out object filter,
                                           [MarshalAs(UnmanagedType.Interface)] out object sink);
        [PreserveSig] int FindInterface([In] ref Guid category, [In] ref Guid type,
                                       IBaseFilter filter, [In] ref Guid iid,
                                       [MarshalAs(UnmanagedType.IUnknown)] out object found);
    }
    [ComImport, Guid("C6E13340-30AC-11D0-A18C-00A0C9118956"), InterfaceType(ComInterfaceType.InterfaceIsIUnknown)]
    interface IAMStreamConfig {
        [PreserveSig] int SetFormat(IntPtr mediaType);
        [PreserveSig] int GetFormat(out IntPtr mediaType);
        [PreserveSig] int GetNumberOfCapabilities(out int count, out int size);
        [PreserveSig] int GetStreamCaps(int index, out IntPtr mediaType, IntPtr capabilities);
    }
    [ComImport, Guid("55272A00-42CB-11CE-8135-00AA004BB851"), InterfaceType(ComInterfaceType.InterfaceIsIUnknown)]
    interface IPropertyBag {
        [PreserveSig] int Read([MarshalAs(UnmanagedType.LPWStr)] string name,
                               [MarshalAs(UnmanagedType.Struct)] ref object value, IntPtr error);
        [PreserveSig] int Write([MarshalAs(UnmanagedType.LPWStr)] string name,
                                [MarshalAs(UnmanagedType.Struct)] ref object value);
    }
    [ComImport, Guid("C6E13360-30AC-11D0-A18C-00A0C9118956"), InterfaceType(ComInterfaceType.InterfaceIsIUnknown)]
    interface IAMVideoProcAmp {
        [PreserveSig] int GetRange(int property, out int min, out int max, out int step, out int defaultValue, out int flags);
        [PreserveSig] int Set(int property, int value, int flags);
        [PreserveSig] int Get(int property, out int value, out int flags);
    }
    [ComImport, Guid("C6E13370-30AC-11D0-A18C-00A0C9118956"), InterfaceType(ComInterfaceType.InterfaceIsIUnknown)]
    interface IAMCameraControl {
        [PreserveSig] int GetRange(int property, out int min, out int max, out int step, out int defaultValue, out int flags);
        [PreserveSig] int Set(int property, int value, int flags);
        [PreserveSig] int Get(int property, out int value, out int flags);
    }
    [StructLayout(LayoutKind.Sequential)] struct Rect { public int Left, Top, Right, Bottom; }
    [StructLayout(LayoutKind.Sequential)] struct Size { public int Width, Height; }
    [StructLayout(LayoutKind.Sequential)] struct BitmapInfo {
        public uint Size; public int Width, Height; public short Planes, BitCount;
        public uint Compression, SizeImage; public int XPels, YPels; public uint ClrUsed, ClrImportant;
    }
    [StructLayout(LayoutKind.Sequential)] struct VideoInfo {
        public Rect Source, Target; public uint BitRate, BitErrorRate; public long AvgTimePerFrame;
        public BitmapInfo Bitmap;
    }
    [StructLayout(LayoutKind.Sequential)] struct VideoInfo2 {
        public Rect Source, Target; public uint BitRate, BitErrorRate; public long AvgTimePerFrame;
        public uint InterlaceFlags, CopyProtectFlags, AspectX, AspectY, ControlFlags, Reserved;
        public BitmapInfo Bitmap;
    }
    [StructLayout(LayoutKind.Sequential)] struct MediaType {
        public Guid MajorType, Subtype;
        public int FixedSizeSamples, TemporalCompression; public uint SampleSize;
        public Guid FormatType; public IntPtr Unknown; public uint FormatSize; public IntPtr Format;
    }
    [StructLayout(LayoutKind.Sequential)] struct VideoCaps {
        public Guid Guid; public uint VideoStandard;
        public Size InputSize, MinCroppingSize, MaxCroppingSize;
        public int CropGranularityX, CropGranularityY, CropAlignX, CropAlignY;
        public Size MinOutputSize, MaxOutputSize;
        public int OutputGranularityX, OutputGranularityY, StretchTapsX, StretchTapsY, ShrinkTapsX, ShrinkTapsY;
        public long MinFrameInterval, MaxFrameInterval;
        public int MinBitsPerSecond, MaxBitsPerSecond;
    }
    public class Mode {
        public string DeviceName {get;set;}
        public int DeviceIndex {get;set;}
        public int Width {get;set;}
        public int Height {get;set;}
        public string FourCC {get;set;}
        public double DefaultFPS {get;set;}
        public double MaxFPS {get;set;}
        public double MinFPS {get;set;}
    }
    public class Control {
        public string Kind {get;set;}
        public string Name {get;set;}
        public int Property {get;set;}
        public int Minimum {get;set;}
        public int Maximum {get;set;}
        public int Step {get;set;}
        public int Default {get;set;}
        public int SupportedFlags {get;set;}
        public int Value {get;set;}
        public int Flags {get;set;}
    }
    public class ControlRequest {
        public string Kind {get;set;}
        public int Property {get;set;}
        public int Value {get;set;}
        public int Flags {get;set;}
    }
    public class ControlState {
        public string DevicePath {get;set;}
        public Control[] Controls {get;set;}
    }
    public static class Modes {
        static Guid VideoCategory = new Guid("860BB310-5D01-11D0-BD3B-00A0C911CE86");
        static Guid CapturePin = new Guid("FB6C4281-0353-11D1-905F-0000C0CC16BA");
        static Guid VideoType = new Guid("73646976-0000-0010-8000-00AA00389B71");
        static Guid ConfigIID = new Guid("C6E13340-30AC-11D0-A18C-00A0C9118956");
        static Guid FilterIID = new Guid("56A86895-0AD4-11CE-B03A-0020AF0BA770");
        static Guid PropertyIID = new Guid("55272A00-42CB-11CE-8135-00AA004BB851");
        static Guid VideoInfoGUID = new Guid("05589F80-C356-11CE-BF01-00AA0055595A");
        static Guid VideoInfo2GUID = new Guid("F72A76A0-EB0A-11D0-ACE4-0000C0CC16BA");
        static void Check(int result) { if (result < 0) Marshal.ThrowExceptionForHR(result); }
        static void Release(object value) { if (value != null && Marshal.IsComObject(value)) Marshal.ReleaseComObject(value); }
        public static ControlState Controls(int deviceIndex, string expectedId, ControlRequest[] requests) {
            object enumObject = null, filter = null;
            IEnumMoniker enumerator = null; IMoniker moniker = null;
            try {
                enumObject = Activator.CreateInstance(Type.GetTypeFromCLSID(new Guid("62BE5D10-60EB-11D0-BD3B-00A0C911CE86")));
                int hr = ((ICreateDevEnum)enumObject).CreateClassEnumerator(ref VideoCategory, out enumerator, 0);
                if (hr != 0 || enumerator == null) throw new InvalidOperationException("No DirectShow camera enumerated");
                var items = new IMoniker[1];
                for (int i = 0; i <= deviceIndex; ++i) {
                    if (enumerator.Next(1, items, IntPtr.Zero) != 0) throw new ArgumentOutOfRangeException("deviceIndex");
                    Release(moniker); moniker = items[0];
                }
                string path;
                moniker.GetDisplayName(null, null, out path);
                if (!String.IsNullOrEmpty(expectedId)) {
                    var parts = expectedId.Split(':');
                    string identity = "vid_" + parts[0] + "&pid_" + parts[1];
                    if (path.IndexOf(identity, StringComparison.OrdinalIgnoreCase) < 0)
                        throw new InvalidOperationException("Camera USB identity differs from profile: " + path);
                }
                moniker.BindToObject(null, null, ref FilterIID, out filter);
                var video = filter as IAMVideoProcAmp;
                var camera = filter as IAMCameraControl;
                foreach (var request in requests ?? new ControlRequest[0]) {
                    int result;
                    if (request.Kind == "video" && video != null) result = video.Set(request.Property, request.Value, request.Flags);
                    else if (request.Kind == "camera" && camera != null) result = camera.Set(request.Property, request.Value, request.Flags);
                    else throw new InvalidOperationException("Requested control interface is unavailable");
                    Check(result);
                }
                string[] videoNames = {"brightness","contrast","hue","saturation","sharpness","gamma","color_enable","white_balance","backlight","gain"};
                string[] cameraNames = {"pan","tilt","roll","zoom","exposure","iris","focus"};
                var controls = new List<Control>();
                for (int kind = 0; kind < 2; ++kind) {
                    string[] names = kind == 0 ? videoNames : cameraNames;
                    for (int i = 0; i < names.Length; ++i) {
                        int min=0,max=0,step=0,def=0,supported=0,value=0,flags=0;
                        int rangeResult=-1, valueResult=-1;
                        if (kind == 0 && video != null) {
                            rangeResult=video.GetRange(i,out min,out max,out step,out def,out supported);
                            valueResult=video.Get(i,out value,out flags);
                        } else if (kind == 1 && camera != null) {
                            rangeResult=camera.GetRange(i,out min,out max,out step,out def,out supported);
                            valueResult=camera.Get(i,out value,out flags);
                        }
                        if (rangeResult >= 0 && valueResult >= 0) controls.Add(new Control {
                            Kind=kind == 0 ? "video" : "camera", Name=names[i], Property=i,
                            Minimum=min,Maximum=max,Step=step,Default=def,SupportedFlags=supported,Value=value,Flags=flags
                        });
                    }
                }
                foreach (var request in requests ?? new ControlRequest[0]) {
                    var actual = controls.Find(c => c.Kind == request.Kind && c.Property == request.Property);
                    // Automatic mode can change the value while we read it back.
                    if (actual == null || actual.Flags != request.Flags ||
                        (request.Flags == 2 && actual.Value != request.Value))
                        throw new InvalidOperationException("Camera control did not retain requested mode/value");
                }
                return new ControlState { DevicePath=path, Controls=controls.ToArray() };
            } finally {
                Release(filter); Release(moniker); Release(enumerator); Release(enumObject);
            }
        }
        public static Mode[] Read(int deviceIndex) {
            var result = new List<Mode>();
            object enumObject = null, builderObject = null, graph = null, filter = null, found = null, bagObject = null;
            IEnumMoniker enumerator = null; IMoniker moniker = null;
            try {
                enumObject = Activator.CreateInstance(Type.GetTypeFromCLSID(new Guid("62BE5D10-60EB-11D0-BD3B-00A0C911CE86")));
                int hr = ((ICreateDevEnum)enumObject).CreateClassEnumerator(ref VideoCategory, out enumerator, 0);
                if (hr != 0 || enumerator == null) throw new InvalidOperationException("No DirectShow camera enumerated");
                var items = new IMoniker[1];
                for (int i = 0; i <= deviceIndex; ++i) {
                    if (enumerator.Next(1, items, IntPtr.Zero) != 0) throw new ArgumentOutOfRangeException("deviceIndex");
                    if (moniker != null) Release(moniker);
                    moniker = items[0];
                }
                moniker.BindToStorage(null, null, ref PropertyIID, out bagObject);
                object deviceName = "";
                Check(((IPropertyBag)bagObject).Read("FriendlyName", ref deviceName, IntPtr.Zero));
                moniker.BindToObject(null, null, ref FilterIID, out filter);
                graph = Activator.CreateInstance(Type.GetTypeFromCLSID(new Guid("E436EBB3-524F-11CE-9F53-0020AF0BA770")));
                builderObject = Activator.CreateInstance(Type.GetTypeFromCLSID(new Guid("BF87B6E1-8C27-11D0-B3F0-00AA003761C5")));
                var builder = (ICaptureGraphBuilder2)builderObject;
                Check(builder.SetFiltergraph((IGraphBuilder)graph));
                Check(builder.FindInterface(ref CapturePin, ref VideoType, (IBaseFilter)filter, ref ConfigIID, out found));
                var config = (IAMStreamConfig)found;
                int count, size;
                Check(config.GetNumberOfCapabilities(out count, out size));
                IntPtr caps = Marshal.AllocCoTaskMem(size);
                try {
                    for (int i = 0; i < count; ++i) {
                        IntPtr mediaPtr = IntPtr.Zero;
                        try {
                            Check(config.GetStreamCaps(i, out mediaPtr, caps));
                            var media = Marshal.PtrToStructure<MediaType>(mediaPtr);
                            BitmapInfo bitmap; long interval;
                            if (media.FormatType == VideoInfoGUID) {
                                var info = Marshal.PtrToStructure<VideoInfo>(media.Format);
                                bitmap = info.Bitmap; interval = info.AvgTimePerFrame;
                            } else if (media.FormatType == VideoInfo2GUID) {
                                var info = Marshal.PtrToStructure<VideoInfo2>(media.Format);
                                bitmap = info.Bitmap; interval = info.AvgTimePerFrame;
                            } else continue;
                            var vc = size >= Marshal.SizeOf<VideoCaps>() ? Marshal.PtrToStructure<VideoCaps>(caps) : default(VideoCaps);
                            byte[] bytes = media.Subtype.ToByteArray();
                            string fourcc = System.Text.Encoding.ASCII.GetString(bytes, 0, 4);
                            result.Add(new Mode {
                                DeviceName = Convert.ToString(deviceName), DeviceIndex = deviceIndex,
                                Width = bitmap.Width, Height = Math.Abs(bitmap.Height), FourCC = fourcc,
                                DefaultFPS = interval > 0 ? 10000000.0 / interval : 0,
                                MaxFPS = vc.MinFrameInterval > 0 ? 10000000.0 / vc.MinFrameInterval : 0,
                                MinFPS = vc.MaxFrameInterval > 0 ? 10000000.0 / vc.MaxFrameInterval : 0
                            });
                        } finally {
                            if (mediaPtr != IntPtr.Zero) {
                                var media = Marshal.PtrToStructure<MediaType>(mediaPtr);
                                if (media.Format != IntPtr.Zero) Marshal.FreeCoTaskMem(media.Format);
                                if (media.Unknown != IntPtr.Zero) Marshal.Release(media.Unknown);
                                Marshal.FreeCoTaskMem(mediaPtr);
                            }
                        }
                    }
                } finally { Marshal.FreeCoTaskMem(caps); }
            } finally {
                Release(found); Release(builderObject); Release(graph); Release(filter);
                Release(bagObject); Release(moniker); Release(enumerator); Release(enumObject);
            }
            return result.ToArray();
        }
    }
}
