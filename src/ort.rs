//! onnxruntime through its C API, over FFI. No crate.
//!
//! `OrtGetApiBase()` returns a table (`OrtApi`) of function pointers. The table
//! layout is append-only and versioned, so this module asks for API version 17
//! and calls only slots at or below index 130. The slot numbers below are the
//! positions of each function in `struct OrtApi` in `onnxruntime_c_api.h`
//! (counted from `CreateStatus` = 0) and are checked at runtime by loading and
//! running the real model in `tests/model.rs`.
//!
//! The library is linked with `#[link(name = "onnxruntime")]`, so the linker
//! needs `libonnxruntime.so` (`LIBRARY_PATH`) to build and the loader needs it
//! (`LD_LIBRARY_PATH` or the system path) to run. Building with
//! `--no-default-features` drops the link and leaves a stub, which lets the
//! pure-logic tests run on a machine without the library.

/// A borrowed f32 output tensor of a model run.
#[derive(Debug, Clone, Copy)]
pub struct Tensor<'a> {
    /// Dimensions, e.g. `[1, 116, 8400]`.
    pub shape: &'a [i64],
    /// Row-major values, valid only inside the `run` callback.
    pub data: &'a [f32],
}

#[cfg(feature = "onnxruntime")]
pub use imp::Session;
#[cfg(not(feature = "onnxruntime"))]
pub use stub::Session;

#[cfg(not(feature = "onnxruntime"))]
mod stub {
    use super::Tensor;
    use std::path::Path;

    /// Placeholder when built without the `onnxruntime` feature.
    #[derive(Debug)]
    pub struct Session;

    impl Session {
        /// Always fails: the binary was built without onnxruntime.
        pub fn load(_model: &Path, _threads: usize) -> Result<Session, String> {
            Err("built without the `onnxruntime` feature".to_string())
        }

        /// Always fails: see [`Session::load`].
        pub fn run<R>(&self, _input: &[f32], _shape: &[i64], _visit: impl FnOnce(&[Tensor<'_>]) -> R) -> Result<R, String> {
            Err("built without the `onnxruntime` feature".to_string())
        }

        /// Library version string; `"none"` without the feature.
        pub fn version() -> String {
            "none".to_string()
        }
    }
}

#[cfg(feature = "onnxruntime")]
mod imp {
    use super::Tensor;
    use std::ffi::{c_char, c_int, c_void, CStr, CString};
    use std::os::unix::ffi::OsStrExt;
    use std::path::Path;
    use std::ptr;

    /// `ORT_API_VERSION` asked of the library. Every slot used here exists at 17.
    const REQUESTED_API_VERSION: u32 = 17;
    /// `ORT_LOGGING_LEVEL_ERROR`.
    const LOG_LEVEL_ERROR: c_int = 3;
    /// `ORT_ENABLE_ALL`.
    const GRAPH_OPT_ALL: c_int = 99;
    /// `OrtArenaAllocator`.
    const ARENA_ALLOCATOR: c_int = 1;
    /// `OrtMemTypeDefault`.
    const MEM_TYPE_DEFAULT: c_int = 0;
    /// `ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT`.
    const ELEMENT_FLOAT: c_int = 1;

    // Slot numbers in `struct OrtApi`.
    const GET_ERROR_MESSAGE: usize = 2;
    const CREATE_ENV: usize = 3;
    const CREATE_SESSION: usize = 7;
    const RUN: usize = 9;
    const CREATE_SESSION_OPTIONS: usize = 10;
    const SET_GRAPH_OPTIMIZATION_LEVEL: usize = 23;
    const SET_INTRA_OP_NUM_THREADS: usize = 24;
    const SET_INTER_OP_NUM_THREADS: usize = 25;
    const SESSION_GET_INPUT_COUNT: usize = 30;
    const SESSION_GET_OUTPUT_COUNT: usize = 31;
    const SESSION_GET_INPUT_NAME: usize = 36;
    const SESSION_GET_OUTPUT_NAME: usize = 37;
    const CREATE_TENSOR_WITH_DATA: usize = 49;
    const GET_TENSOR_MUTABLE_DATA: usize = 51;
    const GET_TENSOR_ELEMENT_TYPE: usize = 60;
    const GET_DIMENSIONS_COUNT: usize = 61;
    const GET_DIMENSIONS: usize = 62;
    const GET_TENSOR_TYPE_AND_SHAPE: usize = 65;
    const CREATE_CPU_MEMORY_INFO: usize = 69;
    const ALLOCATOR_FREE: usize = 76;
    const GET_ALLOCATOR_WITH_DEFAULT_OPTIONS: usize = 78;
    const RELEASE_ENV: usize = 92;
    const RELEASE_STATUS: usize = 93;
    const RELEASE_MEMORY_INFO: usize = 94;
    const RELEASE_SESSION: usize = 95;
    const RELEASE_VALUE: usize = 96;
    const RELEASE_TENSOR_TYPE_AND_SHAPE_INFO: usize = 99;
    const RELEASE_SESSION_OPTIONS: usize = 100;

    macro_rules! opaque {
        ($($name:ident),*) => { $( #[repr(C)] struct $name { _private: [u8; 0] } )* };
    }
    opaque!(OrtEnv, OrtStatus, OrtSessionOptions, OrtSession, OrtMemoryInfo, OrtValue, OrtTensorTypeAndShapeInfo, OrtAllocator, OrtRunOptions);

    #[repr(C)]
    struct OrtApiBase {
        get_api: unsafe extern "C" fn(u32) -> *const c_void,
        get_version_string: unsafe extern "C" fn() -> *const c_char,
    }

    #[link(name = "onnxruntime")]
    extern "C" {
        fn OrtGetApiBase() -> *const OrtApiBase;
    }

    type Status = *mut OrtStatus;
    type ErrorMessageFn = unsafe extern "C" fn(*const OrtStatus) -> *const c_char;
    type CreateEnvFn = unsafe extern "C" fn(c_int, *const c_char, *mut *mut OrtEnv) -> Status;
    type CreateOptionsFn = unsafe extern "C" fn(*mut *mut OrtSessionOptions) -> Status;
    type SetIntFn = unsafe extern "C" fn(*mut OrtSessionOptions, c_int) -> Status;
    type CreateSessionFn = unsafe extern "C" fn(*const OrtEnv, *const c_char, *const OrtSessionOptions, *mut *mut OrtSession) -> Status;
    type CountFn = unsafe extern "C" fn(*const OrtSession, *mut usize) -> Status;
    type NameFn = unsafe extern "C" fn(*const OrtSession, usize, *mut OrtAllocator, *mut *mut c_char) -> Status;
    type GetAllocatorFn = unsafe extern "C" fn(*mut *mut OrtAllocator) -> Status;
    type AllocatorFreeFn = unsafe extern "C" fn(*mut OrtAllocator, *mut c_void) -> Status;
    type CreateMemoryInfoFn = unsafe extern "C" fn(c_int, c_int, *mut *mut OrtMemoryInfo) -> Status;
    type CreateTensorFn = unsafe extern "C" fn(*const OrtMemoryInfo, *mut c_void, usize, *const i64, usize, c_int, *mut *mut OrtValue) -> Status;
    type RunFn = unsafe extern "C" fn(
        *mut OrtSession,
        *const OrtRunOptions,
        *const *const c_char,
        *const *const OrtValue,
        usize,
        *const *const c_char,
        usize,
        *mut *mut OrtValue,
    ) -> Status;
    type TensorDataFn = unsafe extern "C" fn(*mut OrtValue, *mut *mut c_void) -> Status;
    type TypeAndShapeFn = unsafe extern "C" fn(*const OrtValue, *mut *mut OrtTensorTypeAndShapeInfo) -> Status;
    type DimCountFn = unsafe extern "C" fn(*const OrtTensorTypeAndShapeInfo, *mut usize) -> Status;
    type DimsFn = unsafe extern "C" fn(*const OrtTensorTypeAndShapeInfo, *mut i64, usize) -> Status;
    type ElementTypeFn = unsafe extern "C" fn(*const OrtTensorTypeAndShapeInfo, *mut c_int) -> Status;
    type ReleaseFn<T> = unsafe extern "C" fn(*mut T);

    /// The `OrtApi` function table. Copyable: it only holds the table pointer.
    #[derive(Clone, Copy)]
    struct Api {
        table: *const *const c_void,
    }

    impl Api {
        /// Function pointer at `slot`.
        ///
        /// # Safety
        /// `T` must be the exact `extern "C"` signature of the function in that slot.
        unsafe fn get<T: Copy>(&self, slot: usize) -> T {
            debug_assert_eq!(std::mem::size_of::<T>(), std::mem::size_of::<*const c_void>());
            std::mem::transmute_copy(&*self.table.add(slot))
        }

        /// Convert an `OrtStatus*` into a `Result`, releasing the status.
        fn check(&self, status: Status) -> Result<(), String> {
            if status.is_null() {
                return Ok(());
            }
            // SAFETY: `status` is a live OrtStatus returned by the library; the
            // message pointer is valid until the status is released.
            unsafe {
                let msg = self.get::<ErrorMessageFn>(GET_ERROR_MESSAGE)(status);
                let text = if msg.is_null() {
                    "unknown onnxruntime error".to_string()
                } else {
                    CStr::from_ptr(msg).to_string_lossy().into_owned()
                };
                self.get::<ReleaseFn<OrtStatus>>(RELEASE_STATUS)(status);
                Err(text)
            }
        }
    }

    /// An ORT object released through its `Release*` slot when dropped.
    struct Owned<T> {
        api: Api,
        ptr: *mut T,
        release_slot: usize,
    }

    impl<T> Owned<T> {
        fn new(api: Api, ptr: *mut T, release_slot: usize) -> Self {
            Owned { api, ptr, release_slot }
        }
    }

    impl<T> Drop for Owned<T> {
        fn drop(&mut self) {
            if !self.ptr.is_null() {
                // SAFETY: `ptr` came from the matching Create call and is released once.
                unsafe { self.api.get::<ReleaseFn<T>>(self.release_slot)(self.ptr) }
            }
        }
    }

    /// A loaded model plus the names needed to run it.
    ///
    /// Field order matters: the session must drop before the environment.
    pub struct Session {
        session: Owned<OrtSession>,
        mem_info: Owned<OrtMemoryInfo>,
        _env: Owned<OrtEnv>,
        api: Api,
        input_name: CString,
        output_names: Vec<CString>,
    }

    // SAFETY: an OrtSession may be used from several threads (`Run` is
    // thread-safe) and the raw pointers are only released in `Drop`.
    unsafe impl Send for Session {}
    unsafe impl Sync for Session {}

    impl std::fmt::Debug for Session {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            f.debug_struct("Session").field("input", &self.input_name).field("outputs", &self.output_names).finish()
        }
    }

    impl Session {
        /// Library version string, e.g. `"1.30.0"`.
        pub fn version() -> String {
            // SAFETY: OrtGetApiBase returns a static table; the version string is static.
            unsafe {
                let base = OrtGetApiBase();
                if base.is_null() {
                    return "unknown".to_string();
                }
                let v = ((*base).get_version_string)();
                if v.is_null() {
                    "unknown".to_string()
                } else {
                    CStr::from_ptr(v).to_string_lossy().into_owned()
                }
            }
        }

        /// Load `model` and prepare it for CPU inference with `threads` intra-op threads.
        ///
        /// # Errors
        /// A message when the library is too old, the model cannot be read, or it
        /// does not have exactly one input and at least one output.
        pub fn load(model: &Path, threads: usize) -> Result<Session, String> {
            let path = CString::new(model.as_os_str().as_bytes()).map_err(|_| "model path contains a NUL byte".to_string())?;
            // SAFETY: every call below matches the C signature of its slot; every
            // out-pointer is a valid local; objects are wrapped in `Owned` right
            // after creation so they are released on any early return.
            unsafe {
                let base = OrtGetApiBase();
                if base.is_null() {
                    return Err("OrtGetApiBase returned null".to_string());
                }
                let table = ((*base).get_api)(REQUESTED_API_VERSION) as *const *const c_void;
                if table.is_null() {
                    return Err(format!(
                        "onnxruntime {} does not support C API version {REQUESTED_API_VERSION}",
                        Session::version()
                    ));
                }
                let api = Api { table };

                let logid = CString::new("yolo-server").map_err(|e| e.to_string())?;
                let mut env = ptr::null_mut();
                api.check(api.get::<CreateEnvFn>(CREATE_ENV)(LOG_LEVEL_ERROR, logid.as_ptr(), &mut env))?;
                let env = Owned::new(api, env, RELEASE_ENV);

                let mut opts = ptr::null_mut();
                api.check(api.get::<CreateOptionsFn>(CREATE_SESSION_OPTIONS)(&mut opts))?;
                let opts = Owned::new(api, opts, RELEASE_SESSION_OPTIONS);
                let threads = threads.clamp(1, 256) as c_int;
                api.check(api.get::<SetIntFn>(SET_INTRA_OP_NUM_THREADS)(opts.ptr, threads))?;
                api.check(api.get::<SetIntFn>(SET_INTER_OP_NUM_THREADS)(opts.ptr, 1))?;
                api.check(api.get::<SetIntFn>(SET_GRAPH_OPTIMIZATION_LEVEL)(opts.ptr, GRAPH_OPT_ALL))?;

                let mut session = ptr::null_mut();
                api.check(api.get::<CreateSessionFn>(CREATE_SESSION)(env.ptr, path.as_ptr(), opts.ptr, &mut session))?;
                let session = Owned::new(api, session, RELEASE_SESSION);

                let (mut n_in, mut n_out) = (0usize, 0usize);
                api.check(api.get::<CountFn>(SESSION_GET_INPUT_COUNT)(session.ptr, &mut n_in))?;
                api.check(api.get::<CountFn>(SESSION_GET_OUTPUT_COUNT)(session.ptr, &mut n_out))?;
                if n_in != 1 || n_out == 0 {
                    return Err(format!("model has {n_in} inputs and {n_out} outputs; expected 1 and at least 1"));
                }
                let mut allocator = ptr::null_mut();
                api.check(api.get::<GetAllocatorFn>(GET_ALLOCATOR_WITH_DEFAULT_OPTIONS)(&mut allocator))?;
                let input_name = read_name(&api, SESSION_GET_INPUT_NAME, session.ptr, 0, allocator)?;
                let output_names = (0..n_out)
                    .map(|i| read_name(&api, SESSION_GET_OUTPUT_NAME, session.ptr, i, allocator))
                    .collect::<Result<Vec<_>, _>>()?;

                let mut mem = ptr::null_mut();
                api.check(api.get::<CreateMemoryInfoFn>(CREATE_CPU_MEMORY_INFO)(ARENA_ALLOCATOR, MEM_TYPE_DEFAULT, &mut mem))?;
                let mem_info = Owned::new(api, mem, RELEASE_MEMORY_INFO);

                Ok(Session { session, mem_info, _env: env, api, input_name, output_names })
            }
        }

        /// Run the model on one f32 input tensor and hand every output to `visit`.
        ///
        /// The output slices borrow memory owned by the runtime and are only valid
        /// inside `visit`; copy what you need to keep.
        ///
        /// # Errors
        /// When the input length does not match `shape`, the runtime rejects the
        /// run, or an output is not an f32 tensor.
        pub fn run<R>(&self, input: &[f32], shape: &[i64], visit: impl FnOnce(&[Tensor<'_>]) -> R) -> Result<R, String> {
            let expected: i64 = shape.iter().product();
            if expected < 0 || input.len() as i64 != expected {
                return Err(format!("input has {} values but shape {shape:?} needs {expected}", input.len()));
            }
            let api = self.api;
            // SAFETY: the input buffer outlives the OrtValue (dropped before return);
            // out-pointers are valid locals; output OrtValues are owned and released
            // after `visit` returns, and the slices handed to it do not escape.
            unsafe {
                let mut in_val = ptr::null_mut();
                api.check(api.get::<CreateTensorFn>(CREATE_TENSOR_WITH_DATA)(
                    self.mem_info.ptr,
                    input.as_ptr() as *mut c_void,
                    std::mem::size_of_val(input),
                    shape.as_ptr(),
                    shape.len(),
                    ELEMENT_FLOAT,
                    &mut in_val,
                ))?;
                let in_val = Owned::new(api, in_val, RELEASE_VALUE);

                let in_names = [self.input_name.as_ptr()];
                let out_names: Vec<*const c_char> = self.output_names.iter().map(|n| n.as_ptr()).collect();
                let inputs = [in_val.ptr as *const OrtValue];
                let mut raw_outs: Vec<*mut OrtValue> = vec![ptr::null_mut(); out_names.len()];
                api.check(api.get::<RunFn>(RUN)(
                    self.session.ptr,
                    ptr::null(),
                    in_names.as_ptr(),
                    inputs.as_ptr(),
                    1,
                    out_names.as_ptr(),
                    out_names.len(),
                    raw_outs.as_mut_ptr(),
                ))?;
                let outs: Vec<Owned<OrtValue>> = raw_outs.into_iter().map(|p| Owned::new(api, p, RELEASE_VALUE)).collect();

                let shapes = outs.iter().map(|o| tensor_shape(&api, o.ptr)).collect::<Result<Vec<_>, _>>()?;
                let mut tensors = Vec::with_capacity(outs.len());
                for (o, shape) in outs.iter().zip(&shapes) {
                    let len: i64 = shape.iter().product();
                    let mut data = ptr::null_mut();
                    api.check(api.get::<TensorDataFn>(GET_TENSOR_MUTABLE_DATA)(o.ptr, &mut data))?;
                    if data.is_null() || len < 0 {
                        return Err("model output has no data".to_string());
                    }
                    tensors.push(Tensor { shape, data: std::slice::from_raw_parts(data as *const f32, len as usize) });
                }
                Ok(visit(&tensors))
            }
        }
    }

    /// Copy an input or output name out of runtime-allocated memory.
    ///
    /// # Safety
    /// `session` and `allocator` must be live, and `slot` must be one of the
    /// `SessionGet{Input,Output}Name` slots.
    unsafe fn read_name(api: &Api, slot: usize, session: *mut OrtSession, index: usize, allocator: *mut OrtAllocator) -> Result<CString, String> {
        let mut raw: *mut c_char = ptr::null_mut();
        api.check(api.get::<NameFn>(slot)(session, index, allocator, &mut raw))?;
        if raw.is_null() {
            return Err("model has an unnamed input or output".to_string());
        }
        let name = CStr::from_ptr(raw).to_owned();
        api.check(api.get::<AllocatorFreeFn>(ALLOCATOR_FREE)(allocator, raw as *mut c_void))?;
        Ok(name)
    }

    /// Shape of an f32 tensor value.
    ///
    /// # Safety
    /// `value` must be a live tensor OrtValue.
    unsafe fn tensor_shape(api: &Api, value: *mut OrtValue) -> Result<Vec<i64>, String> {
        let mut info = ptr::null_mut();
        api.check(api.get::<TypeAndShapeFn>(GET_TENSOR_TYPE_AND_SHAPE)(value, &mut info))?;
        let info = Owned::new(*api, info, RELEASE_TENSOR_TYPE_AND_SHAPE_INFO);
        let mut element = 0;
        api.check(api.get::<ElementTypeFn>(GET_TENSOR_ELEMENT_TYPE)(info.ptr, &mut element))?;
        if element != ELEMENT_FLOAT {
            return Err(format!("model output element type {element} is not f32"));
        }
        let mut rank = 0usize;
        api.check(api.get::<DimCountFn>(GET_DIMENSIONS_COUNT)(info.ptr, &mut rank))?;
        let mut dims = vec![0i64; rank];
        api.check(api.get::<DimsFn>(GET_DIMENSIONS)(info.ptr, dims.as_mut_ptr(), rank))?;
        Ok(dims)
    }
}
