import * as THREE from 'three';

const SCHEMA = '1.0';
const EPS = 1e-7;
const v3 = () => new THREE.Vector3();
const finite = value => typeof value === 'number' && Number.isFinite(value);
const array = vector => vector.toArray().map(value => Object.is(value, -0) ? 0 : value);

function assert(condition, message) {
  if (!condition) throw new Error(`Harness contract: ${message}`);
}

function effectiveVisible(object, camera) {
  for (let node = object; node; node = node.parent) if (!node.visible) return false;
  if (!object.layers.test(camera.layers)) return false;
  const materials = Array.isArray(object.material) ? object.material : [object.material];
  return materials.some(m => m && m.visible && (!m.transparent || m.opacity > 0));
}

function assertMeasurable(mesh) {
  assert(!mesh.isSkinnedMesh && !mesh.isBatchedMesh,
    'skinned/batched geometry is unsupported in v1; use rigid Mesh primitives.');
  assert(mesh.geometry?.getAttribute('position'), 'each mesh needs a position buffer.');
  assert(mesh.geometry.drawRange.start === 0 && mesh.geometry.drawRange.count === Infinity,
    'partial draw ranges are unsupported in v1; create a complete geometry instead.');
  assert(Object.keys(mesh.geometry.morphAttributes || {}).length === 0,
    'morph targets are unsupported in v1; use explicit rigid transforms.');
  assert(!Array.isArray(mesh.material), 'v1 measures one material per mesh; split multi-material geometry into registered meshes.');
  const materials = [mesh.material];
  for (const mat of materials) {
    assert(mat && !mat.isShaderMaterial && !mat.isRawShaderMaterial && !mat.displacementMap,
      'custom shader/displacement geometry cannot be measured reliably; use a standard material.');
    assert(mat.onBeforeCompile === THREE.Material.prototype.onBeforeCompile,
      'onBeforeCompile deformation is unsupported; use unmodified built-in materials.');
  }
}

function meshesUnder(object) {
  const result = [];
  object.traverse(node => { if (node.isMesh) result.push(node); });
  return result;
}

function isDescendant(object, root) {
  for (let node = object; node; node = node.parent) if (node === root) return true;
  return false;
}

// Verify the position and topology buffers, not just a user-written geometry.type.
function canonicalShape(geometry) {
  const p = geometry.parameters;
  let reference, shape;
  if (geometry instanceof THREE.BoxGeometry && p) {
    if (![p.width, p.height, p.depth].every(n => finite(n) && n > 0)) return null;
    reference = new THREE.BoxGeometry(p.width, p.height, p.depth, p.widthSegments, p.heightSegments, p.depthSegments);
    shape = { type: 'obb', half: [p.width / 2, p.height / 2, p.depth / 2] };
  } else if (geometry instanceof THREE.SphereGeometry && p) {
    if (!(finite(p.radius) && p.radius > 0 && Math.abs(p.phiLength - Math.PI * 2) < EPS
      && Math.abs(p.thetaLength - Math.PI) < EPS && Math.abs(p.thetaStart) < EPS)) return null;
    reference = new THREE.SphereGeometry(p.radius, p.widthSegments, p.heightSegments, p.phiStart, p.phiLength, p.thetaStart, p.thetaLength);
    shape = { type: 'sphere', radius: p.radius };
  } else return null;
  const a = geometry.getAttribute('position'), b = reference.getAttribute('position');
  let equal = a.count === b.count && a.itemSize === b.itemSize;
  for (let i = 0; equal && i < a.count; i++) {
    equal = a.getX(i) === b.getX(i) && a.getY(i) === b.getY(i) && a.getZ(i) === b.getZ(i);
  }
  const ai = geometry.getIndex(), bi = reference.getIndex();
  equal &&= Boolean(ai) === Boolean(bi);
  if (equal && ai) {
    equal &&= ai.count === bi.count;
    for (let i = 0; equal && i < ai.count; i++) equal = ai.getX(i) === bi.getX(i);
  }
  reference.dispose();
  // A partial drawRange is not the full primitive.
  equal &&= geometry.drawRange.start === 0 && geometry.drawRange.count === Infinity;
  return equal ? shape : null;
}

function exactCollider(meshes) {
  if (meshes.length !== 1 || meshes[0].isInstancedMesh) return { collider: null, reason: 'composite_or_instanced_subject' };
  const mesh = meshes[0], shape = canonicalShape(mesh.geometry);
  if (!shape) return { collider: null, reason: 'not_an_unmodified_full_sphere_or_box' };
  const e = mesh.matrixWorld.elements;
  const columns = [new THREE.Vector3(e[0], e[1], e[2]), new THREE.Vector3(e[4], e[5], e[6]), new THREE.Vector3(e[8], e[9], e[10])];
  const scales = columns.map(c => c.length());
  if (!scales.every(s => finite(s) && s > EPS)) return { collider: null, reason: 'singular_transform' };
  const axes = columns.map((c, i) => c.divideScalar(scales[i]));
  if ([axes[0].dot(axes[1]), axes[0].dot(axes[2]), axes[1].dot(axes[2])].some(d => Math.abs(d) > 1e-6)) {
    return { collider: null, reason: 'sheared_world_transform' };
  }
  const center = array(new THREE.Vector3().setFromMatrixPosition(mesh.matrixWorld));
  if (shape.type === 'sphere') {
    if (Math.max(...scales) - Math.min(...scales) > 1e-6 * Math.max(...scales)) {
      return { collider: null, reason: 'nonuniform_sphere_scale' };
    }
    return { collider: { type: 'sphere', center, radius: shape.radius * scales[0], exact: true }, reason: null };
  }
  return { collider: { type: 'obb', center, half_extents: shape.half.map((h, i) => h * scales[i]), axes: axes.map(array), exact: true }, reason: null };
}

function countInstances(mesh) {
  return mesh.isInstancedMesh ? mesh.count : 1;
}

function measureSubject(subject, camera, frustum) {
  const meshes = meshesUnder(subject.object);
  assert(meshes.length > 0, `subject ${subject.id} contains no mesh.`);
  const bounds = new THREE.Box3();
  const visibleBounds = new THREE.Box3();
  const ndcMin = new THREE.Vector2(Infinity, Infinity), ndcMax = new THREE.Vector2(-Infinity, -Infinity);
  const cameraPoint = v3(), worldPoint = v3(), localPoint = v3();
  const instanceMatrix = new THREE.Matrix4(), worldMatrix = new THREE.Matrix4();
  let minDepth = Infinity, maxDepth = -Infinity, fullMinDepth = Infinity, fullMaxDepth = -Infinity;
  let count = 0, projectedCount = 0, triangleCount = 0, visibleMeshCount = 0;
  for (const mesh of meshes) {
    assertMeasurable(mesh);
    const position = mesh.geometry.getAttribute('position');
    const index = mesh.geometry.getIndex();
    const renderedVertexCount = index ? index.count : position.count;
    const n = countInstances(mesh);
    const isVisible = effectiveVisible(mesh, camera);
    if (isVisible) visibleMeshCount++;
    triangleCount += n * (mesh.geometry.index ? mesh.geometry.index.count : position.count) / 3;
    for (let instance = 0; instance < n; instance++) {
      if (mesh.isInstancedMesh) {
        mesh.getMatrixAt(instance, instanceMatrix);
        worldMatrix.multiplyMatrices(mesh.matrixWorld, instanceMatrix);
      } else worldMatrix.copy(mesh.matrixWorld);
      for (let i = 0; i < renderedVertexCount; i++) {
        const vertex = index ? index.getX(i) : i;
        assert(Number.isInteger(vertex) && vertex >= 0 && vertex < position.count,
          `subject ${subject.id} contains an invalid vertex index.`);
        localPoint.fromBufferAttribute(position, vertex);
        assert(array(localPoint).every(finite), `subject ${subject.id} contains nonfinite vertices.`);
        worldPoint.copy(localPoint).applyMatrix4(worldMatrix);
        assert(array(worldPoint).every(finite), `subject ${subject.id} has nonfinite world coordinates.`);
        bounds.expandByPoint(worldPoint);
        if (isVisible) visibleBounds.expandByPoint(worldPoint);
        cameraPoint.copy(worldPoint).applyMatrix4(camera.matrixWorldInverse);
        const depth = -cameraPoint.z;
        fullMinDepth = Math.min(fullMinDepth, depth);
        fullMaxDepth = Math.max(fullMaxDepth, depth);
        count++;
        if (!isVisible) continue;
        minDepth = Math.min(minDepth, depth);
        maxDepth = Math.max(maxDepth, depth);
        projectedCount++;
        if (depth > 0) {
          cameraPoint.applyMatrix4(camera.projectionMatrix);
          ndcMin.x = Math.min(ndcMin.x, cameraPoint.x);
          ndcMin.y = Math.min(ndcMin.y, cameraPoint.y);
          ndcMax.x = Math.max(ndcMax.x, cameraPoint.x);
          ndcMax.y = Math.max(ndcMax.y, cameraPoint.y);
        }
      }
    }
  }
  assert(count > 0 && !bounds.isEmpty(), `subject ${subject.id} has no measurable vertices.`);
  // Conservative validity policy: no projected claim if ANY geometry crosses a clip plane.
  const projectionValid = projectedCount > 0 && minDepth >= camera.near && maxDepth <= camera.far
    && [...ndcMin.toArray(), ...ndcMax.toArray()].every(finite);
  const inFront = fullMinDepth > 0;
  const frustumIntersects = frustum.intersectsBox(bounds);
  const full = projectionValid && ndcMin.x >= -1 - EPS && ndcMin.y >= -1 - EPS
    && ndcMax.x <= 1 + EPS && ndcMax.y <= 1 + EPS;
  let area = null;
  if (projectionValid) {
    const w = Math.max(0, Math.min(1, ndcMax.x) - Math.max(-1, ndcMin.x));
    const h = Math.max(0, Math.min(1, ndcMax.y) - Math.max(-1, ndcMin.y));
    area = w * h / 4;
  }
  const collision = exactCollider(meshes);
  return {
    id: subject.id, role: subject.role || 'primary',
    world_aabb: { min: array(bounds.min), max: array(bounds.max) },
    frustum_intersects: frustumIntersects, in_front: inFront, fully_in_frame: full,
    frustum_intersection_semantics: 'conservative_world_aabb_test',
    projection_valid: projectionValid,
    projection_reason: projectionValid ? null : (projectedCount === 0 ? 'no_visible_geometry' : minDepth <= 0 ? 'camera_plane_or_behind' : minDepth < camera.near ? 'near_plane_crossing' : maxDepth > camera.far ? 'far_plane_crossing' : 'nonfinite_projection'),
    projected_area_ratio: area,
    ndc_box: projectionValid ? { min: ndcMin.toArray(), max: ndcMax.toArray() } : null,
    visible: visibleMeshCount > 0 && fullMaxDepth > camera.near && !visibleBounds.isEmpty() && frustum.intersectsBox(visibleBounds),
    visibility_semantics: 'render_flags_and_frustum_only_not_an_occlusion_test',
    projected_area_semantics: 'viewport_clipped_projected_vertex_bbox_not_pixel_mask_area',
    collider: collision.collider, collider_reason: collision.reason,
    primitive_count: meshes.reduce((sum, mesh) => sum + countInstances(mesh), 0),
    mesh_count: meshes.length, vertex_count: count, triangle_count: triangleCount,
  };
}

/** Create protected rendering/measurement hooks. `update(t)` must be absolute and deterministic. */
export function createHarness({ scene, camera, renderer, update, subjects, seed = 42 }) {
  assert(scene?.isScene, 'scene must be a THREE.Scene.');
  assert(camera?.isPerspectiveCamera, 'v1 requires a PerspectiveCamera.');
  assert(renderer?.isWebGLRenderer, 'renderer must be a WebGLRenderer.');
  assert(typeof update === 'function', 'update(t) must be a function.');
  assert(Array.isArray(subjects) && subjects.length > 0, 'register nonempty subjects.');
  assert(renderer.domElement.id === 'scene', 'use canvas id="scene".');
  assert(Number.isSafeInteger(seed), 'seed must be a safe integer.');
  const ids = new Set();
  const registrations = subjects.map(subject => {
    assert(typeof subject.id === 'string' && subject.id.length > 0 && !ids.has(subject.id), 'subject IDs must be unique nonempty strings.');
    assert(subject.object?.isObject3D && isDescendant(subject.object, scene), `subject ${subject.id} must belong to the scene.`);
    ids.add(subject.id);
    return Object.freeze({ id: subject.id, role: subject.role || 'primary', object: subject.object });
  });
  renderer.setPixelRatio(1);
  renderer.setAnimationLoop(null);
  let lastTime = null;
  function prepare(t) {
    assert(finite(t) && t >= 0, 'time must be a finite nonnegative number.');
    update(t);
    assert(camera.near > 0 && camera.far > camera.near && camera.fov > 0 && camera.fov < 180, 'invalid perspective camera clipping planes or field of view.');
    scene.updateMatrixWorld(true);
    camera.updateProjectionMatrix();
    camera.updateMatrixWorld(true);
    lastTime = t;
  }
  function renderAt(t) {
    prepare(t);
    renderer.setRenderTarget(null);
    renderer.render(scene, camera);
    return { t, width: renderer.domElement.width, height: renderer.domElement.height };
  }
  function exportManifest(t) {
    prepare(t);
    const allMeshes = meshesUnder(scene);
    const registered = new Set();
    for (const subject of registrations) {
      assert(isDescendant(subject.object, scene), `subject ${subject.id} was removed from the scene.`);
      for (const mesh of meshesUnder(subject.object)) {
        assert(!registered.has(mesh), 'subject registrations overlap; register each mesh once.');
        registered.add(mesh);
      }
    }
    const frustum = new THREE.Frustum().setFromProjectionMatrix(new THREE.Matrix4().multiplyMatrices(camera.projectionMatrix, camera.matrixWorldInverse));
    const forward = camera.getWorldDirection(v3()).normalize();
    return {
      schema_version: SCHEMA, t,
      camera: { position: array(camera.getWorldPosition(v3())), quaternion: array(camera.getWorldQuaternion(new THREE.Quaternion())),
        forward: array(forward), pitch_deg: THREE.MathUtils.radToDeg(Math.asin(THREE.MathUtils.clamp(forward.y, -1, 1))),
        pitch_convention: 'signed_forward_elevation_from_xz_plane_down_is_negative', fov_deg: camera.getEffectiveFOV(), near: camera.near, far: camera.far },
      objects: registrations.map(subject => measureSubject(subject, camera, frustum)),
      audit: { mesh_count: allMeshes.length, registered_mesh_count: registered.size,
        unregistered_mesh_count: allMeshes.length - registered.size,
        primitive_count: allMeshes.reduce((sum, mesh) => sum + countInstances(mesh), 0) },
    };
  }
  function renderTopAt(t) {
    const previousTime = lastTime;
    prepare(t);
    const focus = new THREE.Box3();
    for (const subject of registrations) {
      if (subject.role !== 'ground' && subject.role !== 'decoration') focus.union(new THREE.Box3().setFromObject(subject.object, true));
    }
    if (focus.isEmpty()) focus.set(new THREE.Vector3(-2, 0, -2), new THREE.Vector3(2, 2, 2));
    const center = focus.getCenter(v3()), span = focus.getSize(v3());
    const aspect = renderer.domElement.width / renderer.domElement.height;
    const halfZ = Math.max(span.z / 2, span.x / (2 * aspect), 1.5) * 1.2;
    const top = new THREE.OrthographicCamera(-halfZ * aspect, halfZ * aspect, halfZ, -halfZ, 0.01, 1000);
    top.position.set(center.x, focus.max.y + Math.max(10, span.y + 5), center.z);
    top.up.set(0, 0, -1);
    top.lookAt(center);
    top.updateMatrixWorld(true);
    renderer.setRenderTarget(null);
    renderer.render(scene, top);
    // Restore scene/camera transforms after the alternate render; canvas keeps the top view.
    if (previousTime !== null && previousTime !== t) prepare(previousTime);
    else lastTime = previousTime;
    return { t, view: 'top', horizontal_axis: '+x_right', vertical_axis: '-z_up', camera_position: array(top.position) };
  }
  const gl = renderer.getContext();
  const debug = gl.getExtension('WEBGL_debug_renderer_info');
  const unmaskedRenderer = debug ? gl.getParameter(debug.UNMASKED_RENDERER_WEBGL) : null;
  const meta = Object.freeze({ schema_version: SCHEMA, three_revision: THREE.REVISION, seed,
    backend: unmaskedRenderer || gl.getParameter(gl.RENDERER), gl_renderer: gl.getParameter(gl.RENDERER),
    unmasked_renderer: unmaskedRenderer,
    unmasked_vendor: debug ? gl.getParameter(debug.UNMASKED_VENDOR_WEBGL) : null,
    measurement: 'geometry_buffers_world_transforms',
    projection_policy: 'invalid_if_any_vertex_crosses_near_far_or_camera_plane',
    collider_policy: 'canonical_full_sphere_or_box_only_no_shear',
  });
  for (const [name, value] of Object.entries({ renderAt, exportManifest, renderTopAt, __harnessMeta: meta, __harnessReady: true })) {
    assert(!Object.prototype.hasOwnProperty.call(window, name), `${name} already exists; do not override capture hooks.`);
    Object.defineProperty(window, name, { value, writable: false, configurable: false, enumerable: true });
  }
  return Object.freeze({ renderAt, exportManifest, renderTopAt, meta });
}
