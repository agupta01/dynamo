// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! HTTP frontend extensions: system GET routes and the opt-in Decisions inference route.

use std::collections::HashSet;
use std::sync::Arc;

use crate::protocols::openai::pooling::{
    NvCreatePoolingRequest, PoolingEmbedDType, PoolingEncodingFormat, PoolingEndianness,
    PoolingInput, PoolingOutput,
};
use axum::Router;
use axum::handler::Handler;
use axum::http::Method;
use axum::http::{HeaderMap, StatusCode};
use axum::routing::{get, post};
use base64::Engine as _;

use super::RouteDoc;
use super::service_v2::State;

/// Live view of frontend state exposed to extensions (typed
/// accessors only, not the internal `State`/`ModelManager`).
#[derive(Clone)]
pub struct FrontendExtensionContext {
    state: Arc<State>,
}

impl FrontendExtensionContext {
    pub(crate) fn new(state: Arc<State>) -> Self {
        Self { state }
    }

    pub fn is_ready(&self) -> bool {
        self.state.is_ready()
    }

    pub fn is_cancelled(&self) -> bool {
        self.state.is_cancelled()
    }

    pub fn has_any_ready_model(&self) -> bool {
        self.state.manager().has_any_ready_model()
    }

    pub fn is_model_ready_to_serve(&self, model: &str) -> bool {
        self.state.manager().is_model_ready_to_serve(model)
    }

    pub fn model_display_names(&self) -> HashSet<String> {
        self.state.manager().model_display_names()
    }

    pub fn serving_ready_display_names(&self) -> HashSet<String> {
        self.state.manager().serving_ready_display_names()
    }

    /// Run one pinned Decider v1.2 sequence through the existing pooling engine.
    /// Decode only the marker rows, not the full token-by-class matrix.
    pub async fn decision_marker_logits(
        &self,
        model: &str,
        ids: Vec<u32>,
        markers: &[usize],
    ) -> Result<Vec<Vec<f32>>, StatusCode> {
        const WIDTH: usize = 255;
        // Each packed float32 row divides evenly into base64 triplets.
        const ROW_BYTES: usize = WIDTH * 4;
        const ROW_CHARS: usize = ROW_BYTES / 3 * 4;
        let n = ids.len();
        if !(1..=32768).contains(&n)
            || !(1..=64).contains(&markers.len())
            || markers.iter().any(|&m| m >= n)
        {
            return Err(StatusCode::BAD_REQUEST);
        }
        let req = NvCreatePoolingRequest {
            model: model.into(),
            input: PoolingInput::Tokens(ids),
            task: Some("token_classify".into()),
            use_activation: Some(false),
            encoding_format: PoolingEncodingFormat::Bytes,
            embed_dtype: Some(PoolingEmbedDType::Float32),
            endianness: Some(PoolingEndianness::Little),
            add_special_tokens: Some(false),
            user: None,
            request_id: None,
            priority: 0,
            mm_processor_kwargs: None,
            cache_salt: None,
            truncation_side: None,
            dimensions: None,
            truncate_prompt_tokens: None,
            nvext: None,
        };
        let (mut response, mut inflight) =
            super::openai::execute_pooling(self.state.clone(), HeaderMap::new(), req)
                .await
                .map_err(|(status, _)| status)?;
        let [item] = response.data.as_mut_slice() else {
            return Err(StatusCode::BAD_GATEWAY);
        };
        if item.index != 0 || item.shape.as_deref() != Some(&[n as u64, WIDTH as u64][..]) {
            return Err(StatusCode::BAD_GATEWAY);
        }
        let PoolingOutput::Base64(data) = &item.data else {
            return Err(StatusCode::BAD_GATEWAY);
        };
        if data.len() != n * ROW_CHARS {
            return Err(StatusCode::BAD_GATEWAY);
        }
        let mut rows = Vec::with_capacity(markers.len());
        let mut decoded = [0; ROW_BYTES];
        for &marker in markers {
            let start = marker * ROW_CHARS;
            let written = base64::engine::general_purpose::STANDARD
                .decode_slice(&data.as_bytes()[start..start + ROW_CHARS], &mut decoded)
                .map_err(|_| StatusCode::BAD_GATEWAY)?;
            if written != ROW_BYTES {
                return Err(StatusCode::BAD_GATEWAY);
            }
            rows.push(
                decoded
                    .chunks_exact(4)
                    .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                    .collect(),
            );
        }
        inflight.mark_ok();
        Ok(rows)
    }
}

/// Routes returned by an extension. Each constructor records matching route docs.
pub struct FrontendRouteSet {
    route_docs: Vec<RouteDoc>,
    router: Router,
}

impl FrontendRouteSet {
    /// Start building a system route set. Only `GET` routes are supported.
    pub fn builder() -> FrontendRouteSetBuilder {
        FrontendRouteSetBuilder::default()
    }

    /// Build an inference route set separately from the system GET routes.
    pub fn decisions<H, T>(handler: H) -> Self
    where
        H: Handler<T, ()>,
        T: 'static,
    {
        const PATH: &str = "/v1/decisions";
        Self {
            route_docs: vec![RouteDoc::new(Method::POST, PATH)],
            router: Router::new().route(PATH, post(handler)),
        }
    }

    pub fn route_docs(&self) -> &[RouteDoc] {
        &self.route_docs
    }

    pub(crate) fn into_parts(self) -> (Vec<RouteDoc>, Router) {
        (self.route_docs, self.router)
    }
}

/// Reject paths that would panic axum's `Router::route` (params, wildcards,
/// `:`/`*` segments, whitespace). Single source of truth, reused by Python.
pub fn validate_extension_route_path(path: &str) -> anyhow::Result<()> {
    if !path.starts_with('/') {
        anyhow::bail!("frontend route path must start with '/': {path:?}");
    }
    // matchit (axum's router) reserves segments starting with ':' or '*'.
    if path.split('/').any(|seg| seg.starts_with([':', '*'])) {
        anyhow::bail!("frontend route path segment must not start with ':' or '*': {path:?}");
    }
    if path
        .chars()
        .any(|c| matches!(c, '{' | '}' | '*') || c.is_whitespace() || c.is_control())
    {
        anyhow::bail!(
            "frontend route path must be static (no '{{...}}', '*', or whitespace): {path:?}"
        );
    }
    Ok(())
}

/// Registers each route into both the router and its docs atomically; `get`
/// validates the path and rejects duplicates instead of panicking axum.
#[derive(Default)]
pub struct FrontendRouteSetBuilder {
    route_docs: Vec<RouteDoc>,
    router: Router,
    seen_paths: HashSet<String>,
}

impl FrontendRouteSetBuilder {
    /// Register a `GET` route, recording it in both the router and the docs.
    /// Errors on an invalid path or a duplicate path within this set.
    pub fn get<H, T>(mut self, path: impl Into<String>, handler: H) -> anyhow::Result<Self>
    where
        H: Handler<T, ()>,
        T: 'static,
    {
        let path = path.into();
        validate_extension_route_path(&path)?;
        if !self.seen_paths.insert(path.clone()) {
            anyhow::bail!("duplicate frontend route registered: GET {path}");
        }
        self.route_docs
            .push(RouteDoc::new(Method::GET, path.clone()));
        self.router = self.router.route(&path, get(handler));
        Ok(self)
    }

    pub fn build(self) -> FrontendRouteSet {
        FrontendRouteSet {
            route_docs: self.route_docs,
            router: self.router,
        }
    }
}

/// Callback that attaches additional frontend routes during HTTP service build,
/// given a read-only [`FrontendExtensionContext`]. Returns an error so invalid
/// extension configuration becomes a clean startup failure.
pub type FrontendRouteExtension = Arc<
    dyn Fn(FrontendExtensionContext) -> anyhow::Result<FrontendRouteSet> + Send + Sync + 'static,
>;

#[cfg(test)]
mod tests {
    use super::*;
    use axum::http::StatusCode;

    #[test]
    fn validate_route_path_rejects_non_static() {
        assert!(validate_extension_route_path("/ok").is_ok());
        assert!(validate_extension_route_path("/a/b_c").is_ok());
        for bad in [
            "no-slash",
            "/:id",
            "/{id}",
            "/x/{*rest}",
            "/*rest",
            "/has space",
        ] {
            assert!(
                validate_extension_route_path(bad).is_err(),
                "expected {bad:?} to be rejected"
            );
        }
    }

    #[test]
    fn builder_rejects_duplicate_path_within_set() {
        let dup = FrontendRouteSet::builder()
            .get("/a", || async { StatusCode::OK })
            .unwrap()
            .get("/a", || async { StatusCode::OK });
        assert!(dup.is_err(), "duplicate GET path within a set must error");
    }

    #[test]
    fn builder_rejects_colon_path() {
        let bad = FrontendRouteSet::builder().get("/:id", || async { StatusCode::OK });
        assert!(bad.is_err(), "colon-prefixed segment must error, not panic");
    }
}
