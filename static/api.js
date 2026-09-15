/**
 * api.js -- thin fetch wrapper over the Virtual Curtain Try-On backend.
 * Now with authentication support.
 */

const API_BASE_URL = window.CURTAIN_API_BASE_URL || "http://172.16.10.18:8000";

class ApiError extends Error {
  constructor(message, status, detail) {
    super(message);
    this.name = "ApiError";
    this.status = status;
    this.detail = detail;
  }
}

function getAuthToken() {
  return localStorage.getItem("access_token");
}

function setAuthToken(token) {
  if (token) localStorage.setItem("access_token", token);
  else localStorage.removeItem("access_token");
}

function getCurrentUser() {
  const userJson = localStorage.getItem("user");
  try { return userJson ? JSON.parse(userJson) : null; } catch { return null; }
}

function setCurrentUser(user) {
  if (user) localStorage.setItem("user", JSON.stringify(user));
  else localStorage.removeItem("user");
}

async function _fetchWithAuth(url, options = {}) {
  const token = getAuthToken();
  const headers = {
    "Content-Type": "application/json",
    ...(options.headers || {}),
  };
  if (token) {
    headers["Authorization"] = `Bearer ${token}`;
  }
  if (options.body instanceof FormData) {
    delete headers["Content-Type"];
  }
  const res = await fetch(url, {
    ...options,
    headers,
  });
  return _handleResponse(res);
}

async function _handleResponse(res) {
  if (!res.ok) {
    let detail = null;
    try {
      const body = await res.json();
      detail = body.detail || body.error || JSON.stringify(body);
    } catch (_) {
      detail = res.statusText;
    }
    throw new ApiError(`Request failed (${res.status})`, res.status, detail);
  }
  if (res.status === 204) return null;
  return res.json();
}

async function registerUser(name, email, password) {
  const res = await fetch(`${API_BASE_URL}/api/auth/register`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ name, email, password }),
  });
  const data = await _handleResponse(res);
  setAuthToken(null);
  setCurrentUser(null);
  return data;
}

async function loginUser(email, password, googleId = "0") {
  const payload = { email, password, google_id: googleId };
  const res = await fetch(`${API_BASE_URL}/api/auth/login`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
  const data = await _handleResponse(res);
  if (data.access_token) {
    setAuthToken(data.access_token);
    const user = await getMe();
    setCurrentUser(user);
    return user;
  }
  throw new ApiError(data.message || "Login failed", 401, data.message);
}

async function getMe() {
  return _fetchWithAuth(`${API_BASE_URL}/api/auth/me`, { method: "GET" });
}

function logoutUser() {
  setAuthToken(null);
  setCurrentUser(null);
  fetch(`${API_BASE_URL}/api/auth/logout`, { method: "POST" }).catch(() => {});
}

async function uploadRoom(file, previousRoomToken = null) {
  const formData = new FormData();
  formData.append("file", file);
  if (previousRoomToken) formData.append("previous_room_token", previousRoomToken);
  return _fetchWithAuth(`${API_BASE_URL}/api/upload-room`, {
    method: "POST",
    body: formData,
  });
}

async function getFabrics() {
  const res = await fetch(`${API_BASE_URL}/api/fabrics`);
  return _handleResponse(res);
}

async function searchFabrics(filters = {}) {
  const res = await fetch(`${API_BASE_URL}/api/search-fabric`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(filters),
  });
  return _handleResponse(res);
}

async function uploadFabric(file, metadata = {}) {
  const formData = new FormData();
  formData.append("file", file);
  formData.append("name", metadata.name || file.name);
  formData.append("pattern", metadata.pattern || "custom");
  formData.append("color", metadata.color || "unknown");
  return _fetchWithAuth(`${API_BASE_URL}/api/upload-fabric`, {
    method: "POST",
    body: formData,
  });
}

async function renderCurtain(roomToken, fabricId, curtainIds = null, patternControls = {}) {
  return _fetchWithAuth(`${API_BASE_URL}/api/render-curtain`, {
    method: "POST",
    body: JSON.stringify({
      room_token: roomToken,
      fabric_id: fabricId,
      curtain_ids: curtainIds,
      pattern_scale: patternControls.scale ?? 1.0,
      pattern_rotation_deg: patternControls.rotationDeg ?? 0.0,
      pattern_offset_x: patternControls.offsetX ?? 0.0,
      pattern_offset_y: patternControls.offsetY ?? 0.0,
      color_hue_shift_deg: patternControls.hueShiftDeg ?? 0.0,
    }),
  });
}

function resolveFabricImageUrl(fabric) {
  if (!fabric || !fabric.texture_image_url) return "";
  if (/^https?:\/\//.test(fabric.texture_image_url)) return fabric.texture_image_url;
  return `${API_BASE_URL}${fabric.texture_image_url}`;
}

async function healthCheck() {
  const res = await fetch(`${API_BASE_URL}/health`);
  return _handleResponse(res);
}

window.CurtainAPI = {
  API_BASE_URL,
  ApiError,
  getAuthToken,
  setAuthToken,
  getCurrentUser,
  setCurrentUser,
  registerUser,
  loginUser,
  getMe,
  logoutUser,
  uploadRoom,
  getFabrics,
  searchFabrics,
  uploadFabric,
  renderCurtain,
  resolveFabricImageUrl,
  healthCheck,
};
