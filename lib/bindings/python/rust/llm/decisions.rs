// SPDX-License-Identifier: Apache-2.0
//! Perplexity Decider v1.2: the stock HTTP server owns the socket and the stock
//! model manager dispatches pooling. Python owns only the checkpoint protocol.
use std::{
    fs::File,
    io::{self, Read},
    path::{Path, PathBuf},
    sync::Arc,
    time::Duration,
};
use subtle::ConstantTimeEq;

use dynamo_llm::http::service::{
    FrontendExtensionContext, FrontendRouteExtension, FrontendRouteSet,
    axum::{
        Json,
        body::{Body, to_bytes},
        http::{HeaderMap, StatusCode, header},
        response::{IntoResponse, Response},
    },
};
use pyo3::{prelude::*, types::PyBytes};
use pythonize::depythonize;
use serde_json::{Value, json};
use tokio::{
    sync::{OwnedSemaphorePermit, Semaphore},
    task::spawn_blocking,
    time::timeout,
};

type Sequence = (Vec<u32>, Vec<usize>);
type Planned = (u16, PyObject, Vec<Sequence>);

// Reopen on every request so an atomic rotation takes effect without restart.
fn read_key(path: &Path) -> io::Result<Vec<u8>> {
    let mut bytes = Vec::new();
    File::open(path)?.take(4098).read_to_end(&mut bytes)?;
    if bytes.ends_with(b"\n") {
        bytes.pop();
    }
    if !(32..=4096).contains(&bytes.len()) || !bytes.iter().all(|b| b.is_ascii_graphic()) {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "invalid Decisions key file",
        ));
    }
    Ok(bytes)
}

pub(super) fn extension(py: Python<'_>, directory: PathBuf) -> PyResult<FrontendRouteExtension> {
    let key_path = Arc::new(PathBuf::from(
        std::env::var_os("DECIDER_API_KEY_FILE").ok_or_else(|| {
            pyo3::exceptions::PyValueError::new_err(
                "set DECIDER_API_KEY_FILE when enabling Decisions",
            )
        })?,
    ));
    read_key(&key_path)?;
    let model: Arc<str> = std::env::var("DYN_DECIDER_SERVED_MODEL")
        .unwrap_or_else(|_| "pplx-decider-v1.2".to_owned())
        .into();
    let codec = Arc::new(
        py.import("dynamo.frontend.decisions_codec")?
            .getattr("Codec")?
            .call1((directory,))?
            .unbind(),
    );
    // One full 32768-token sequence can return ~43 MiB of base64 from the worker.
    // Bound this independently of the stock endpoints or number of questions.
    let slots = Arc::new(Semaphore::new(2));
    Ok(Arc::new(move |context| {
        let codec = Arc::clone(&codec);
        let key_path = Arc::clone(&key_path);
        let model = Arc::clone(&model);
        let slots = Arc::clone(&slots);
        let routes = FrontendRouteSet::decisions(move |headers: HeaderMap, body: Body| {
            let codec = Arc::clone(&codec);
            let key_path = Arc::clone(&key_path);
            let model = Arc::clone(&model);
            let context = context.clone();
            let permit = Arc::clone(&slots).try_acquire_owned();
            async move {
                let Ok(permit) = permit else {
                    return error(StatusCode::SERVICE_UNAVAILABLE, "Frontend busy");
                };
                match timeout(
                    Duration::from_secs(300),
                    serve(
                        codec,
                        key_path,
                        &model,
                        context,
                        headers,
                        body,
                        Arc::new(permit),
                    ),
                )
                .await
                {
                    Ok(Ok(response)) | Ok(Err(response)) => response,
                    Err(_) => error(
                        StatusCode::SERVICE_UNAVAILABLE,
                        "Inference deadline exceeded",
                    ),
                }
            }
        });
        Ok(routes)
    }))
}

fn error(status: StatusCode, message: &str) -> Response {
    let kind = if status.is_server_error() {
        "server_error"
    } else {
        "invalid_request_error"
    };
    (
        status,
        [(header::CACHE_CONTROL, "no-store")],
        Json(json!({
            "error": {"message": message, "type": kind, "param": null, "code": null}
        })),
    )
        .into_response()
}

async fn cpu<F, T>(permit: Arc<OwnedSemaphorePermit>, f: F) -> Result<T, Response>
where
    F: FnOnce(Python<'_>) -> PyResult<T> + Send + 'static,
    T: Send + 'static,
{
    // A disconnected client must not release the CPU slot until tokenization
    // actually finishes; blocking work is not cancellable.
    spawn_blocking(move || {
        let _permit = permit;
        Python::with_gil(f)
    })
    .await
    .ok()
    .and_then(Result::ok)
    .ok_or_else(|| {
        error(
            StatusCode::INTERNAL_SERVER_ERROR,
            "Decision processing failed",
        )
    })
}

async fn serve(
    codec: Arc<PyObject>,
    key_path: Arc<PathBuf>,
    model: &str,
    context: FrontendExtensionContext,
    headers: HeaderMap,
    body: Body,
    permit: Arc<OwnedSemaphorePermit>,
) -> Result<Response, Response> {
    let mut authorization = headers.get_all(header::AUTHORIZATION).iter();
    let token = authorization
        .next()
        .and_then(|v| v.to_str().ok())
        .and_then(|v| v.split_once(' '))
        .filter(|(scheme, token)| {
            scheme.eq_ignore_ascii_case("bearer") && (32..=4096).contains(&token.len())
        })
        .map(|(_, token)| token.as_bytes());
    let Some(token) = token.filter(|_| authorization.next().is_none()) else {
        return Err(error(StatusCode::UNAUTHORIZED, "Invalid API key"));
    };
    let expected = spawn_blocking(move || read_key(&key_path))
        .await
        .ok()
        .and_then(Result::ok)
        .ok_or_else(|| {
            error(
                StatusCode::SERVICE_UNAVAILABLE,
                "Authentication unavailable",
            )
        })?;
    if !bool::from(expected.ct_eq(token)) {
        return Err(error(StatusCode::UNAUTHORIZED, "Invalid API key"));
    }
    let media_type = headers
        .get(header::CONTENT_TYPE)
        .and_then(|v| v.to_str().ok())
        .and_then(|v| v.split(';').next())
        .unwrap_or("")
        .trim();
    if !media_type.eq_ignore_ascii_case("application/json")
        || headers
            .get(header::CONTENT_ENCODING)
            .is_some_and(|v| !v.as_bytes().eq_ignore_ascii_case(b"identity"))
    {
        return Err(error(
            StatusCode::UNSUPPORTED_MEDIA_TYPE,
            "Expected an uncompressed JSON body",
        ));
    }
    let bytes = timeout(Duration::from_secs(30), to_bytes(body, 1024 * 1024))
        .await
        .map_err(|_| error(StatusCode::REQUEST_TIMEOUT, "Request body timeout"))?
        .map_err(|_| error(StatusCode::PAYLOAD_TOO_LARGE, "Request body exceeds 1 MiB"))?;
    let encoder = Arc::clone(&codec);
    let (status, plan, sequences): Planned = cpu(Arc::clone(&permit), move |py| {
        encoder
            .call_method1(py, "plan", (PyBytes::new(py, &bytes),))?
            .extract(py)
    })
    .await?;
    if status != 200 {
        let body: Value = cpu(permit, move |py| Ok(depythonize(plan.bind(py))?)).await?;
        return Err((
            StatusCode::BAD_REQUEST,
            [(header::CACHE_CONTROL, "no-store")],
            Json(body),
        )
            .into_response());
    }
    let mut rows = Vec::new();
    for (ids, markers) in sequences {
        let mut output = context
            .decision_marker_logits(model, ids, &markers)
            .await
            .map_err(|status| error(status, "Pooling worker could not serve the decision"))?;
        rows.append(&mut output);
    }
    let response: Value = cpu(permit, move |py| {
        let result = codec.call_method1(py, "finish", (plan, rows))?;
        Ok(depythonize(result.bind(py))?)
    })
    .await?;
    Ok(([(header::CACHE_CONTROL, "no-store")], Json(response)).into_response())
}
