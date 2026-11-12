/* Characters, clips and items with three.js: the studio's equip preview, and a game's starting point (the game export,
   rigging/export_game.py, copies this file beside its manifest.json and an example page using it).

   A character is one GLB: its meshes, skeleton and socket nodes (socket_hand_R...). Its clips are GLBs of the same
   skeleton's animation, played by bone name. An item is a GLB whose origin is its grip, +Y towards its tip, in metres,
   so it goes under a socket node with no transform of its own.

     import { Viewer, Character, load } from './equip.js';
     const viewer = new Viewer(document.querySelector('#view'));
     const rogue = new Character(await load('characters/rogue/character.glb'));
     rogue.addClips(await load('characters/rogue/clips/walk.glb'));
     rogue.play('walk');
     rogue.equip('socket_hand_R', (await load('items/dagger.glb')).scene);
     viewer.show(rogue);

   The page needs an import map for 'three' and 'three/addons/' (see the example page), and serving over http
   (python3 -m http.server): browsers don't load modules or GLBs from file://.
*/
import * as THREE from 'three';
import { GLTFLoader } from 'three/addons/loaders/GLTFLoader.js';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';

const loader = new GLTFLoader();
const cache = new Map();

/** A GLB, loaded once per URL. */
export function load(url) {
  if (!cache.has(url)) cache.set(url, loader.loadAsync(url).catch(e => { cache.delete(url); throw e; }));
  return cache.get(url);
}

export class Character {
  constructor(gltf) {
    this.root = gltf.scene;
    this.mixer = new THREE.AnimationMixer(this.root);
    this.clips = new Map();
    this.action = null;
    this.items = new Map();
    this.addClips(gltf);
    this.root.traverse(o => {
      if (!o.isMesh) return;
      o.castShadow = true;
      o.frustumCulled = false;  // a skinned mesh's bounds are its rest pose's, not where the clip moves it
    });
  }

  /** The clips of a GLB of this character's skeleton. name renames a GLB's only clip. */
  addClips(gltf, name) {
    for (const c of gltf.animations) {
      const n = name && gltf.animations.length === 1 ? name : c.name;
      if (this.clips.has(n)) continue;
      const clip = n === c.name ? c : c.clone();
      clip.name = n;
      this.clips.set(n, clip);
    }
    return this;
  }

  /** The names of its socket nodes. */
  sockets() {
    const s = [];
    this.root.traverse(o => { if (o.name.startsWith('socket_')) s.push(o.name); });
    return s;
  }

  /** Play a clip, cross-faded from the one playing. */
  play(name, { loop = true, fade = 0.2 } = {}) {
    const clip = this.clips.get(name);
    if (!clip) throw new Error(`no clip named ${name}`);
    const next = this.mixer.clipAction(clip);
    next.reset().setLoop(loop ? THREE.LoopRepeat : THREE.LoopOnce, Infinity);
    next.clampWhenFinished = !loop;
    if (this.action && this.action !== next) next.crossFadeFrom(this.action, fade, false);
    next.play();
    this.action = next;
    return next;
  }

  /** Put an item (a loaded item GLB's scene; a copy of it is used) on a socket, or empty the socket (null). */
  equip(socket, item) {
    const s = this.root.getObjectByName(socket);
    if (!s) throw new Error(`no socket named ${socket}`);
    const old = this.items.get(socket);
    if (old) s.remove(old);
    this.items.delete(socket);
    if (!item) return null;
    const o = item.clone();
    o.traverse(x => { if (x.isMesh) x.castShadow = true; });
    s.add(o);
    this.items.set(socket, o);
    return o;
  }

  update(dt) { this.mixer.update(dt); }
}

export class Viewer {
  constructor(el) {
    this.el = el;
    this.renderer = new THREE.WebGLRenderer({ antialias: true });
    this.renderer.setPixelRatio(Math.min(devicePixelRatio, 2));
    this.renderer.shadowMap.enabled = true;
    el.append(this.renderer.domElement);
    this.scene = new THREE.Scene();
    this.scene.background = new THREE.Color(0x1b1e22);
    this.scene.add(new THREE.HemisphereLight(0xffffff, 0x404040, 2.2));
    const sun = new THREE.DirectionalLight(0xffffff, 2.4);
    sun.position.set(2, 4, 3);
    sun.castShadow = true;
    sun.shadow.mapSize.set(2048, 2048);
    Object.assign(sun.shadow.camera, { left: -2, right: 2, top: 2, bottom: -2 });
    this.scene.add(sun);
    const ground = new THREE.Mesh(new THREE.PlaneGeometry(20, 20), new THREE.ShadowMaterial({ opacity: 0.35 }));
    ground.rotation.x = -Math.PI / 2;
    ground.receiveShadow = true;
    this.scene.add(ground, new THREE.GridHelper(10, 20, 0x3a4048, 0x2a2f36));
    this.camera = new THREE.PerspectiveCamera(35, 1, 0.01, 100);
    this.camera.position.set(1.2, 1.4, 3.6);
    this.controls = new OrbitControls(this.camera, this.renderer.domElement);
    this.controls.target.set(0, 0.9, 0);
    this.characters = [];
    this.clock = new THREE.Clock();
    this.resize = new ResizeObserver(() => {
      const w = el.clientWidth || 1, h = el.clientHeight || 1;
      this.renderer.setSize(w, h);
      this.camera.aspect = w / h;
      this.camera.updateProjectionMatrix();
    });
    this.resize.observe(el);
    this.renderer.setAnimationLoop(() => this.tick());
  }

  /** Show a character (alone), the camera framing it. */
  show(character, { frame = true } = {}) {
    for (const c of this.characters) this.scene.remove(c.root);
    this.characters = [character];
    this.scene.add(character.root);
    if (frame) this.frame(character.root);
  }

  frame(object) {
    const box = new THREE.Box3().setFromObject(object);
    const size = box.getSize(new THREE.Vector3()), centre = box.getCenter(new THREE.Vector3());
    const d = Math.max(size.y, size.x) * 2.1;
    this.controls.target.copy(centre);
    this.camera.position.copy(centre).add(new THREE.Vector3(0.35 * d, 0.15 * d, d));  // characters face +z
    this.controls.update();
  }

  tick() {
    if (!this.el.isConnected) return this.dispose();  // its page moved on
    const dt = this.clock.getDelta();
    for (const c of this.characters) c.update(dt);
    this.controls.update();
    this.renderer.render(this.scene, this.camera);
  }

  dispose() {
    this.renderer.setAnimationLoop(null);
    this.resize.disconnect();
    this.controls.dispose();
    this.renderer.dispose();
    this.renderer.domElement.remove();
  }
}
