import * as THREE from 'three';

/** Small neutral primitives. These helpers never schedule an animation loop. */
export const NEUTRAL = Object.freeze({ object: 0x9b9d9e, ground: 0xe2e1dc, dark: 0x727779 });

export function material(color = NEUTRAL.object) {
  return new THREE.MeshStandardMaterial({ color, roughness: 0.92, metalness: 0 });
}

export function box(size = [1, 1, 1], options = {}) {
  const mesh = new THREE.Mesh(new THREE.BoxGeometry(...size), material(options.color));
  mesh.castShadow = true;
  mesh.receiveShadow = true;
  if (options.position) mesh.position.fromArray(options.position);
  return mesh;
}

export function sphere(radius = 0.5, options = {}) {
  const mesh = new THREE.Mesh(new THREE.SphereGeometry(radius, 32, 20), material(options.color));
  mesh.castShadow = true;
  mesh.receiveShadow = true;
  if (options.position) mesh.position.fromArray(options.position);
  return mesh;
}

export function cylinder(radius = 0.4, height = 1, options = {}) {
  const mesh = new THREE.Mesh(new THREE.CylinderGeometry(radius, radius, height, 32), material(options.color));
  mesh.castShadow = true;
  mesh.receiveShadow = true;
  if (options.position) mesh.position.fromArray(options.position);
  return mesh;
}

export function addNeutralStage(scene, { size = 12, groundY = 0 } = {}) {
  scene.background = new THREE.Color(0xf1f0eb);
  const ground = box([size, 0.08, size], { color: NEUTRAL.ground, position: [0, groundY - 0.04, 0] });
  ground.name = 'ground';
  ground.castShadow = false;
  scene.add(ground);
  scene.add(new THREE.HemisphereLight(0xffffff, 0x8e918d, 2.2));
  const key = new THREE.DirectionalLight(0xffffff, 2.3);
  key.position.set(-3, 8, 5);
  key.castShadow = true;
  key.shadow.mapSize.set(1024, 1024);
  key.shadow.camera.left = -6;
  key.shadow.camera.right = 6;
  key.shadow.camera.top = 6;
  key.shadow.camera.bottom = -6;
  key.shadow.normalBias = 0.025;
  scene.add(key);
  return ground;
}

export function smoothstep01(t) {
  const x = THREE.MathUtils.clamp(t, 0, 1);
  return x * x * (3 - 2 * x);
}

/** Deterministic PRNG for scene construction, not animation-time random draws. */
export function seededRandom(seed = 42) {
  let state = seed >>> 0;
  return () => {
    state += 0x6d2b79f5;
    let n = Math.imul(state ^ (state >>> 15), 1 | state);
    n ^= n + Math.imul(n ^ (n >>> 7), 61 | n);
    return ((n ^ (n >>> 14)) >>> 0) / 4294967296;
  };
}
