"""Is the robot link-footprint metadata frame-aligned with each scene's world?

For several scenes, set the arm to the CANONICAL qpos (dq=0) and compare the
metadata link origin (x0,y0,z0) against the TRUE world body_xpos. A large,
scene-dependent CONSTANT offset == the metadata is in a different world frame
than the scene (robot base placed elsewhere), so the require-graph link
positions are systematically wrong in those scenes.
"""
from __future__ import annotations
import random, sys
import numpy as np
from libero_infinity.compiler import compile_task_to_scenario
from libero_infinity.gym_env import make_env
from libero_infinity.task_config import TaskConfig
from libero_infinity.robot_metadata import get_robot_footprint
from libero_infinity.validation.sweep import resolve_task_path

TASKS = [
    "libero_90/LIVING_ROOM_SCENE1_pick_up_the_cream_cheese_box_and_put_it_in_the_basket.bddl",
    "libero_10/KITCHEN_SCENE3_turn_on_the_stove_and_put_the_moka_pot_on_it.bddl",
    "libero_goal/put_the_wine_bottle_on_the_rack.bddl",
    "libero_goal/put_the_bowl_on_the_stove.bddl",
]

def main():
    import mujoco
    fp = get_robot_footprint("Panda")
    canon = np.array(fp.canonical_qpos)
    print(f"metadata table_world_z={fp.table_world_z:.4f}")
    for task in TASKS:
        bddl = str(resolve_task_path(task))
        cfg = TaskConfig.from_bddl(bddl)
        random.seed(0)
        scn = compile_task_to_scenario(cfg, "robot")
        scene,_ = scn.generate(maxIterations=4000)
        env = make_env(scene, bddl_path=bddl); env.reset()
        sim = env._sim.libero_env.env.sim
        robot = env._sim.libero_env.env.robots[0]
        jidx = np.asarray(robot._ref_joint_pos_indexes, dtype=int)
        mjm,mjd = sim.model._model, sim.data._data
        mjd.qpos[jidx]=canon; mjd.qvel[:]=0; mujoco.mj_forward(mjm,mjd)
        # robot base body world pos
        base_xpos=None
        for b in range(sim.model.nbody):
            bn=sim.model.body_id2name(b)
            if bn=="robot0_base":
                base_xpos=np.array(sim.data.body_xpos[b]); break
        offs=[]
        for lk in fp.active_links():
            for b in range(sim.model.nbody):
                if sim.model.body_id2name(b)==lk.name:
                    true_o=np.array(sim.data.body_xpos[b])
                    meta=np.array([lk.x0,lk.y0,lk.z0])
                    offs.append(true_o-meta); break
        offs=np.array(offs)
        print(f"\n{task.split('/')[-1]}")
        print(f"  robot0_base world xpos = {base_xpos}")
        print(f"  mean(true-meta) offset = {offs.mean(0)}  std={offs.std(0)}  |mean|={np.linalg.norm(offs.mean(0))*1000:.1f}mm")
        env.close()

if __name__=="__main__":
    sys.exit(main())
