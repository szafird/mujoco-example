// Proof-of-concept: MuJoCo 3.5.0 WASM + Three.js, all physics runs in browser.
// External CDN assets require internet access and permissive notebook CSP.
import * as THREE from 'https://esm.sh/three@0.180.0';
import { OrbitControls } from 'https://esm.sh/three@0.180.0/examples/jsm/controls/OrbitControls.js';
import loadMujoco from 'https://esm.sh/@mujoco/mujoco@3.5.0?bundle';

const MJ_WASM = 'https://cdn.jsdelivr.net/npm/@mujoco/mujoco@3.5.0/dist/mujoco.wasm';
let modulePromise;
function getMujoco() {
  modulePromise ||= loadMujoco({locateFile: (path) => path.endsWith('.wasm') ? MJ_WASM : path});
  return modulePromise;
}

export default {
  async render({model, el}) {
    el.innerHTML = '';
    const root = document.createElement('div');
    root.style.cssText = 'width:100%;font:13px system-ui;color:#222';
    const toolbar = document.createElement('div');
    toolbar.style.cssText = 'display:flex;align-items:center;gap:10px;margin-bottom:8px;flex-wrap:wrap';
    const play = document.createElement('button'); play.textContent='Pause';
    const reset = document.createElement('button'); reset.textContent='Reset';
    const status = document.createElement('span'); status.textContent='Loading MuJoCo WASM…';
    const time = document.createElement('span'); time.textContent='t = 0.00 s';
    toolbar.append(play,reset,time,status);
    const viewport = document.createElement('div');
    viewport.style.cssText='height:420px;width:100%;background:#eef3f8;position:relative;overflow:hidden;border-radius:8px';
    root.append(toolbar,viewport); el.append(root);
    let disposed=false, running=true, frame=0, mj=null, m=null, d=null, mjScene=null, opt=null, perturb=null, cameraMJ=null;
    let renderer, controls, scene, camera, observer;
    const meshes=[];
    const setStatus=(s)=>{status.textContent=s; model.set('status',s); model.save_changes();};
    const handleCommand=()=>{
      if(!mj || !m || !d) return;
      let c; try {c=JSON.parse(model.get('command'));} catch {return;}
      if(c.op==='pause') running=false;
      if(c.op==='play') running=true;
      if(c.op==='reset') {mj.mj_resetData(m,d);mj.mj_forward(m,d);}
      if(c.op==='gravity') {m.opt.gravity[2]=c.value;}
      if(c.op==='control' && c.index>=0 && c.index<d.ctrl.length) d.ctrl[c.index]=c.value;
      play.textContent=running?'Pause':'Play';
    };
    model.on('change:command',handleCommand);
    play.onclick=()=>{running=!running;play.textContent=running?'Pause':'Play';};
    reset.onclick=()=>{if(mj&&m&&d){mj.mj_resetData(m,d);mj.mj_forward(m,d);}};
    try {
      mj=await getMujoco(); if(disposed) return;
      m=mj.MjModel.from_xml_string(model.get('xml'));
      d=new mj.MjData(m);
      opt=new mj.MjvOption(); perturb=new mj.MjvPerturb(); cameraMJ=new mj.MjvCamera();
      mjScene=new mj.MjvScene(m,1000);
      scene=new THREE.Scene();scene.background=new THREE.Color(0xeef3f8);
      scene.add(new THREE.HemisphereLight(0xffffff,0x64748b,2.2));
      const light=new THREE.DirectionalLight(0xffffff,2);light.position.set(-3,-4,8);scene.add(light);
      camera=new THREE.PerspectiveCamera(45,1,0.01,100);
      camera.up.set(0,0,1);camera.position.set(2.8,-3.8,2.6);
      renderer=new THREE.WebGLRenderer({antialias:true});renderer.setPixelRatio(Math.min(devicePixelRatio,2));
      viewport.append(renderer.domElement);
      controls=new OrbitControls(camera,renderer.domElement);controls.target.set(0,0,0.5);controls.update();
      const resize=()=>{const w=viewport.clientWidth||700,h=viewport.clientHeight;renderer.setSize(w,h);camera.aspect=w/h;camera.updateProjectionMatrix();};
      observer=new ResizeObserver(resize);observer.observe(viewport);resize();
      setStatus('Ready (drag to orbit, scroll to zoom)');
      let last=performance.now(), accumulator=0, lastSync=0;
      const animate=(now)=>{
        if(disposed) return;
        frame=requestAnimationFrame(animate);
        const dt=Math.min((now-last)/1000,0.05);last=now;
        if(running){
          accumulator+=dt;
          const step=Math.max(0.0001,Number(m.opt.timestep));
          let count=0;
          while(accumulator>=step && count<100){mj.mj_step(m,d);accumulator-=step;count++;}
          if(count===100) accumulator=0;
        } else accumulator=0;
        mj.mjv_updateScene(m,d,opt,perturb,cameraMJ,mj.mjtCatBit.mjCAT_ALL.value,mjScene);
        const geoms=mjScene.geoms;
        const count=Number(geoms.size());
        for(let i=0;i<count;i++){
          const g=geoms.get(i);
          const type=Number(g.type),s=Array.from(g.size),rgba=Array.from(g.rgba),p=Array.from(g.pos),r=Array.from(g.mat);
          let mesh=meshes[i];
          if(!mesh){
            let geometry;
            if(type===mj.mjtGeom.mjGEOM_PLANE.value) geometry=new THREE.PlaneGeometry(s[0]>0?s[0]*2:20,s[1]>0?s[1]*2:20);
            else if(type===mj.mjtGeom.mjGEOM_SPHERE.value) geometry=new THREE.SphereGeometry(s[0],32,20);
            else if(type===mj.mjtGeom.mjGEOM_BOX.value) geometry=new THREE.BoxGeometry(2*s[0],2*s[1],2*s[2]);
            else if(type===mj.mjtGeom.mjGEOM_CYLINDER.value){geometry=new THREE.CylinderGeometry(s[0],s[0],2*s[2],24);geometry.rotateX(Math.PI/2);}
            else if(type===mj.mjtGeom.mjGEOM_CAPSULE.value){geometry=new THREE.CapsuleGeometry(s[0],2*s[2],8,16);geometry.rotateX(Math.PI/2);}
            else geometry=new THREE.SphereGeometry(0.01,8,6);
            mesh=new THREE.Mesh(geometry,new THREE.MeshPhongMaterial({color:new THREE.Color(rgba[0],rgba[1],rgba[2]),transparent:rgba[3]<1,opacity:rgba[3],side:THREE.DoubleSide}));
            scene.add(mesh);meshes[i]=mesh;
          }
          mesh.matrixAutoUpdate=false;
          mesh.matrix.set(r[0],r[1],r[2],p[0],r[3],r[4],r[5],p[1],r[6],r[7],r[8],p[2],0,0,0,1);
          mesh.matrixWorldNeedsUpdate=true;mesh.visible=true;
          g.delete();
        }
        for(let i=count;i<meshes.length;i++) meshes[i].visible=false;
        geoms.delete();
        controls.update();renderer.render(scene,camera);
        time.textContent=`t = ${Number(d.time).toFixed(2)} s`;
        if(now-lastSync>500){lastSync=now;model.set('sim_time',Number(d.time));model.save_changes();}
      };
      frame=requestAnimationFrame(animate);
    }catch(e){console.error(e);setStatus('Error: '+(e?.message||e));}
    return ()=>{
      disposed=true;cancelAnimationFrame(frame);model.off('change:command',handleCommand);
      observer?.disconnect();controls?.dispose();
      for(const mesh of meshes){mesh.geometry.dispose();mesh.material.dispose();}
      renderer?.dispose();mjScene?.delete();cameraMJ?.delete();perturb?.delete();opt?.delete();d?.delete();m?.delete();
    };
  }
};
