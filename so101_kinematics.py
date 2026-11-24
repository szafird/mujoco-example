"""Menagerie SO-101 kinematic teaching helpers.

Loads the upstream MJCF, preserves the joint chain and position actuators,
and replaces STL mesh visualization with simple collision-free primitives for
V4's lightweight browser renderer. Not a visual/CAD-fidelity replacement.
"""
from __future__ import annotations
import copy
import math
import urllib.request
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from typing import Sequence
import numpy as np

MENAGERIE_SO101_URL = (
    'https://raw.githubusercontent.com/google-deepmind/mujoco_menagerie/'
    'main/robotstudio_so101/so101.xml'
)
ARM_JOINTS = ('shoulder_pan', 'shoulder_lift', 'elbow_flex', 'wrist_flex', 'wrist_roll')
ALL_JOINTS = ARM_JOINTS + ('gripper',)


def load_menagerie_xml(url: str = MENAGERIE_SO101_URL) -> str:
    """Download original MJCF (rather than retyping the robot's transforms)."""
    with urllib.request.urlopen(url, timeout=35) as response:
        return response.read().decode('utf-8')


def _vec(text, length=3, default=None):
    if text is None:
        if default is None:
            raise ValueError('Missing expected vector')
        return np.asarray(default, dtype=float)
    values = np.fromstring(text, sep=' ', dtype=float)
    if len(values) != length:
        raise ValueError(f'Expected {length} numbers, got {text!r}')
    return values


def _quat_matrix(q):
    w, x, y, z = np.asarray(q, dtype=float)
    s = w*w + x*x + y*y + z*z
    if s == 0:
        raise ValueError('Invalid zero quaternion')
    return np.array([
        [1 - 2*(y*y+z*z)/s, 2*(x*y-z*w)/s, 2*(x*z+y*w)/s],
        [2*(x*y+z*w)/s, 1 - 2*(x*x+z*z)/s, 2*(y*z-x*w)/s],
        [2*(x*z-y*w)/s, 2*(y*z+x*w)/s, 1 - 2*(x*x+y*y)/s],
    ])


def _pose(pos=None, quat=None):
    T = np.eye(4)
    T[:3, :3] = _quat_matrix(_vec(quat, 4, [1,0,0,0]))
    T[:3, 3] = _vec(pos, 3, [0,0,0])
    return T


def _rotation(axis, angle):
    axis = np.asarray(axis, dtype=float)
    axis = axis / np.linalg.norm(axis)
    x,y,z = axis
    c, s = math.cos(angle), math.sin(angle)
    K = np.array([[0,-z,y],[z,0,-x],[-y,x,0]])
    R = np.eye(4)
    R[:3,:3] = np.eye(3) + s*K + (1-c)*K@K
    return R


def _translation(v):
    T = np.eye(4)
    T[:3,3] = v
    return T


@dataclass
class IKResult:
    q: np.ndarray
    reached: bool
    error: float
    iterations: int
    positions: list


class SO101Kinematics:
    """Position FK and DLS IK using transforms parsed from upstream MJCF."""
    def __init__(self, xml: str):
        self.original_xml = xml
        self.root = ET.fromstring(xml)
        self.worldbody = self.root.find('worldbody')
        if self.worldbody is None:
            raise ValueError('No worldbody in MJCF')
        self.joint_names = ALL_JOINTS
        joints = {j.get('name'):j for j in self.root.iter('joint') if j.get('name')}
        missing = set(ALL_JOINTS) - set(joints)
        if missing:
            raise ValueError(f'Menagerie SO-101 joints missing: {missing}')
        self.limits = np.array([_vec(joints[name].get('range'), 2) for name in ALL_JOINTS])
        self.joint_indices = {name:i for i,name in enumerate(ALL_JOINTS)}
        sites = [s for s in self.root.iter('site') if s.get('name')=='gripperframe']
        if len(sites)!=1:
            raise ValueError('Expected unique gripperframe site')

    def fk(self, q: Sequence[float]) -> np.ndarray:
        """World transform of Menagerie gripperframe site, in metres."""
        q=np.asarray(q,dtype=float)
        if q.shape != (6,):
            raise ValueError('SO-101 expects 6 joint coordinates, including gripper')
        found=[]
        def visit(body, parent):
            T = parent @ _pose(body.get('pos'), body.get('quat'))
            for joint in body.findall('joint'):
                name=joint.get('name')
                if name in self.joint_indices:
                    if joint.get('type','hinge')!='hinge':
                        raise ValueError(f'Unsupported joint type in {name}')
                    p = _vec(joint.get('pos'), 3, [0,0,0])
                    axis = _vec(joint.get('axis'), 3, [0,0,1])
                    angle = q[self.joint_indices[name]] - float(joint.get('ref','0'))
                    T = T @ _translation(p) @ _rotation(axis, angle) @ _translation(-p)
            for site in body.findall('site'):
                if site.get('name')=='gripperframe':
                    found.append(T @ _pose(site.get('pos'),site.get('quat')))
            for child in body.findall('body'):
                visit(child, T)
        for body in self.worldbody.findall('body'):
            visit(body, np.eye(4))
        if len(found)!=1:
            raise ValueError(f'Expected one gripperframe, found {len(found)}')
        return found[0]

    def xyz(self, q: Sequence[float]) -> np.ndarray:
        return self.fk(q)[:3,3]

    def jacobian(self, q: Sequence[float], eps=1e-5) -> np.ndarray:
        """3x5 position Jacobian, evaluated using central differences."""
        q=np.asarray(q,dtype=float)
        J=np.empty((3,5))
        for i in range(5):
            qp, qm = q.copy(), q.copy()
            qp[i] += eps; qm[i] -= eps
            J[:,i] = (self.xyz(qp) - self.xyz(qm))/(2*eps)
        return J

    def ik_position(self, target: Sequence[float], seed=None, max_iter=250,
                    damping=0.035, gain=0.65, tolerance=0.003) -> IKResult:
        """Damped least squares, fixed gripper joint; position only, not orientation."""
        target=np.asarray(target,dtype=float)
        if target.shape!=(3,):
            raise ValueError('target must be xyz')
        q=np.zeros(6) if seed is None else np.asarray(seed,dtype=float).copy()
        q=np.clip(q,self.limits[:,0],self.limits[:,1])
        positions=[self.xyz(q).copy()]
        for i in range(max_iter):
            e=target-positions[-1]
            err=float(np.linalg.norm(e))
            if err<tolerance:
                return IKResult(q, True, err, i, positions)
            J=self.jacobian(q)
            # Damped least squares: J^T (J J^T + lambda^2 I)^-1 e
            dq=J.T @ np.linalg.solve(J@J.T + damping**2*np.eye(3), e)
            q[:5] = np.clip(q[:5] + gain*dq, self.limits[:5,0], self.limits[:5,1])
            positions.append(self.xyz(q).copy())
        return IKResult(q, False, float(np.linalg.norm(target-positions[-1])), max_iter, positions)


def make_teaching_xml(original_xml: str, target=None) -> str:
    """Keep authentic SO-101 joints/inertials/actuators/sites, replace all mesh
    and collision geoms with easily rendered, non-colliding teaching shapes.

    V4 does NOT render STL assets; this is a schematic geometric depiction.
    Robot position and all FK calculations still use original Menagerie frames.
    """
    root=ET.fromstring(original_xml)
    root.set('model', 'SO-101 Menagerie kinematics (primitive visualization)')
    compiler=root.find('compiler')
    if compiler is not None:
        compiler.attrib.pop('meshdir',None)
    for asset in root.findall('asset'):
        for mesh in asset.findall('mesh'):
            asset.remove(mesh)
    world=root.find('worldbody')
    palette=['0.98 0.67 0.15 1','0.98 0.74 0.19 1','0.95 0.48 0.14 1',
             '0.22 0.66 0.85 1','0.25 0.52 0.78 1','0.18 0.28 0.40 1','0.85 0.36 0.22 1']
    bodies=list(world.iter('body'))
    for idx,body in enumerate(bodies):
        # Remove original rendered/collision geometry only. Preserve inertials,
        # joints, axes, all original body transforms, sites and position actuators.
        for geom in list(body.findall('geom')):
            body.remove(geom)
        children=body.findall('body')
        pos=np.array([float(x) for x in body.get('pos','0 0 0').split()])
        color=palette[min(idx,len(palette)-1)]
        ET.SubElement(body,'geom',name=f'schematic_{idx}_hub', type='sphere',
                      size='0.026',rgba=color,contype='0',conaffinity='0',mass='0')
        for j,child in enumerate(children):
            offset=_vec(child.get('pos'),3,[0,0,0])
            distance=float(np.linalg.norm(offset))
            if distance>0.035:
                start=offset*(0.02/distance)
                end=offset*(max(0.025,distance-0.02)/distance)
                ET.SubElement(body,'geom',name=f'schematic_{idx}_{j}',
                              type='capsule',size='0.020',
                              fromto=' '.join(f'{v:.7g}' for v in np.r_[start,end]),
                              rgba=color,contype='0',conaffinity='0',mass='0')
    # A very small, floating position marker marks the named gripper frame.
    # The source site remains unchanged and is used for FK.
    for site in root.iter('site'):
        if site.get('name')=='gripperframe':
            site.set('type','sphere');site.set('size','0.018')
            site.set('rgba','0.13 0.92 0.45 1');site.set('group','0')
    ET.SubElement(world,'geom',name='floor',type='plane',pos='0 0 -0.028',
                  size='1 1 0.1',rgba='0.82 0.88 0.93 1',
                  contype='0',conaffinity='0')
    ET.SubElement(world,'light',pos='0 -1 2')
    if target is not None:
        t=np.asarray(target,dtype=float)
        ET.SubElement(world,'geom',name='ik_target',type='sphere',
                      pos=' '.join(f'{x:.8g}' for x in t),size='0.025',
                      rgba='0.85 0.18 0.67 1',contype='0',conaffinity='0')
    # A keyframe is not used: arm starts at qpos=0; actuators start at ctrl=0.
    return ET.tostring(root,encoding='unicode')


MENAGERIE_SO101_ASSETS = (
    "https://raw.githubusercontent.com/google-deepmind/mujoco_menagerie/"
    "main/robotstudio_so101/assets/"
)


def make_mesh_scene_xml(original_xml: str, target=None) -> str:
    """Return Menagerie's real SO-101 geometry, materials and collision geoms.

    The V5 browser frontend downloads files named by `<asset><mesh file=...>`
    from MENAGERIE_SO101_ASSETS and loads them into MuJoCo's virtual FS.
    `meshdir=assets` from the upstream compiler remains unchanged.
    """
    root = ET.fromstring(original_xml)
    root.set('model', 'SO-101 Menagerie real meshes')
    world = root.find('worldbody')
    if world is None:
        raise ValueError('Expected worldbody')
    ET.SubElement(world, 'light', pos='0 -1 2')
    # Plane is rendered but collision disabled to avoid introducing contact
    # forces into the FK/IK teaching exercises.
    ET.SubElement(world, 'geom', name='display_floor', type='plane',
                  pos='0 0 -0.035', size='1 1 0.1',
                  rgba='0.83 0.89 0.94 1', contype='0', conaffinity='0')
    if target is not None:
        t = np.asarray(target, dtype=float)
        ET.SubElement(world, 'geom', name='ik_target', type='sphere',
                      pos=' '.join(f'{x:.8g}' for x in t), size='0.018',
                      rgba='0.88 0.16 0.63 1', contype='0', conaffinity='0')
    return ET.tostring(root, encoding='unicode')
