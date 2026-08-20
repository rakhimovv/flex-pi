import sys

sys.path.append("./")

import sapien.core as sapien
from sapien.render import clear_cache
from collections import OrderedDict
import pdb
from envs import *
import yaml
import importlib
import json
import traceback
import os
import time
from argparse import ArgumentParser

current_file_path = os.path.abspath(__file__)
parent_directory = os.path.dirname(current_file_path)


def class_decorator(task_name):
    envs_module = importlib.import_module(f"envs.{task_name}")
    try:
        env_class = getattr(envs_module, task_name)
        env_instance = env_class()
    except:
        raise SystemExit("No such task")
    return env_instance


def get_embodiment_config(robot_file):
    robot_config_file = os.path.join(robot_file, "config.yml")
    with open(robot_config_file, "r", encoding="utf-8") as f:
        embodiment_args = yaml.load(f.read(), Loader=yaml.FullLoader)
    return embodiment_args


def main(task_name=None, task_config=None):

    task = class_decorator(task_name)
    config_path = f"./task_config/{task_config}.yml"

    with open(config_path, "r", encoding="utf-8") as f:
        args = yaml.load(f.read(), Loader=yaml.FullLoader)

    args['task_name'] = task_name

    embodiment_type = args.get("embodiment")
    embodiment_config_path = os.path.join(CONFIGS_PATH, "_embodiment_config.yml")

    with open(embodiment_config_path, "r", encoding="utf-8") as f:
        _embodiment_types = yaml.load(f.read(), Loader=yaml.FullLoader)

    def get_embodiment_file(embodiment_type):
        robot_file = _embodiment_types[embodiment_type]["file_path"]
        if robot_file is None:
            raise "missing embodiment files"
        return robot_file

    if len(embodiment_type) == 1:
        args["left_robot_file"] = get_embodiment_file(embodiment_type[0])
        args["right_robot_file"] = get_embodiment_file(embodiment_type[0])
        args["dual_arm_embodied"] = True
    elif len(embodiment_type) == 3:
        args["left_robot_file"] = get_embodiment_file(embodiment_type[0])
        args["right_robot_file"] = get_embodiment_file(embodiment_type[1])
        args["embodiment_dis"] = embodiment_type[2]
        args["dual_arm_embodied"] = False
    else:
        raise "number of embodiment config parameters should be 1 or 3"

    args["left_embodiment_config"] = get_embodiment_config(args["left_robot_file"])
    args["right_embodiment_config"] = get_embodiment_config(args["right_robot_file"])

    if len(embodiment_type) == 1:
        embodiment_name = str(embodiment_type[0])
    else:
        embodiment_name = str(embodiment_type[0]) + "+" + str(embodiment_type[1])

    # show config
    print("============= Config =============\n")
    print("\033[95mMessy Table:\033[0m " + str(args["domain_randomization"]["cluttered_table"]))
    print("\033[95mRandom Background:\033[0m " + str(args["domain_randomization"]["random_background"]))
    if args["domain_randomization"]["random_background"]:
        print(" - Clean Background Rate: " + str(args["domain_randomization"]["clean_background_rate"]))
    print("\033[95mRandom Light:\033[0m " + str(args["domain_randomization"]["random_light"]))
    if args["domain_randomization"]["random_light"]:
        print(" - Crazy Random Light Rate: " + str(args["domain_randomization"]["crazy_random_light_rate"]))
    print("\033[95mRandom Table Height:\033[0m " + str(args["domain_randomization"]["random_table_height"]))
    print("\033[95mRandom Head Camera Distance:\033[0m " + str(args["domain_randomization"]["random_head_camera_dis"]))

    print("\033[94mHead Camera Config:\033[0m " + str(args["camera"]["head_camera_type"]) + f", " +
          str(args["camera"]["collect_head_camera"]))
    print("\033[94mWrist Camera Config:\033[0m " + str(args["camera"]["wrist_camera_type"]) + f", " +
          str(args["camera"]["collect_wrist_camera"]))
    print("\033[94mEmbodiment Config:\033[0m " + embodiment_name)
    print("\n==================================")

    args["embodiment_name"] = embodiment_name
    args['task_config'] = task_config
    args["save_path"] = os.path.join(args["save_path"], str(args["task_name"]), args["task_config"])
    run(task, args)


def run(TASK_ENV, args):
    epid, suc_num, fail_num, seed_list = 0, 0, 0, []

    print(f"Task Name: \033[34m{args['task_name']}\033[0m")

    # =========== Collect Seed ===========
    os.makedirs(args["save_path"], exist_ok=True)

    if not args["use_seed"]:
        print("\033[93m" + "[Start Seed and Pre Motion Data Collection]" + "\033[0m")
        args["need_plan"] = True

        if os.path.exists(os.path.join(args["save_path"], "seed.txt")):
            with open(os.path.join(args["save_path"], "seed.txt"), "r") as file:
                seed_list = file.read().split()
                if len(seed_list) != 0:
                    seed_list = [int(i) for i in seed_list]
                    suc_num = len(seed_list)
                    epid = max(seed_list) + 1
            print(f"Exist seed file, Start from: {epid} / {suc_num}")

        # Backfill _traj_data for seeds already in seed.txt but missing their traj pickle.
        # Replans only the known-good seeds — skips the failed-attempt cost of the main loop.
        # If a seed fails to replan (sim/planner/asset drift, e.g. seed.txt imported from
        # another machine), the seed is dropped and the rest are renumbered to keep
        # _traj_data/episode{i}.pkl indices contiguous. Replacements are added by the
        # main while-loop below, which resumes from the new max(seed)+1.
        # Safety: this drop-and-renumber is only allowed when no episode*.hdf5 exists yet,
        # so we never invalidate previously-collected Phase-2 data.
        if args.get("collect_data", False) and seed_list:
            traj_dir = os.path.join(args["save_path"], "_traj_data")
            data_dir = os.path.join(args["save_path"], "data")
            hdf5_present = os.path.isdir(data_dir) and any(
                f.startswith("episode") and f.endswith(".hdf5") for f in os.listdir(data_dir)
            )
            missing_idx = [
                i for i in range(len(seed_list))
                if not os.path.exists(os.path.join(traj_dir, f"episode{i}.pkl"))
            ]
            if missing_idx:
                print(f"\033[93m[Backfill traj] {len(missing_idx)} / {len(seed_list)} episodes missing traj"
                      f" (hdf5_present={hdf5_present})\033[0m")
                failed_idx = []
                for i in missing_idx:
                    seed = seed_list[i]
                    try:
                        TASK_ENV.setup_demo(now_ep_num=i, seed=seed, **args)
                        TASK_ENV.play_once()
                        ok = TASK_ENV.plan_success and TASK_ENV.check_success()
                        TASK_ENV.close_env()
                        if args["render_freq"]:
                            TASK_ENV.viewer.close()
                    except Exception as e:
                        print(f"backfill episode {i} fail! (seed = {seed})  Error: {e}")
                        ok = False
                        try: TASK_ENV.close_env()
                        except Exception: pass

                    if ok:
                        TASK_ENV.save_traj_data(i)
                        print(f"backfill episode {i} success! (seed = {seed})")
                    else:
                        failed_idx.append(i)
                        if hdf5_present:
                            raise RuntimeError(
                                f"seed {seed} (episode {i}) no longer plans successfully and "
                                f"{data_dir} already contains hdf5 episodes — refusing to "
                                "drop+renumber because that would invalidate previously "
                                "collected Phase-2 data. Delete seed.txt + _traj_data/ + data/ "
                                "and re-run Phase 1 from scratch."
                            )

                if failed_idx:
                    drop = set(failed_idx)
                    print(f"\033[93m[Backfill compact] dropping {len(drop)} unrecoverable seeds: "
                          f"{sorted(drop)}\033[0m")
                    new_seeds, new_idx = [], 0
                    for old_idx, s in enumerate(seed_list):
                        if old_idx in drop:
                            continue
                        old_path = os.path.join(traj_dir, f"episode{old_idx}.pkl")
                        new_path = os.path.join(traj_dir, f"episode{new_idx}.pkl")
                        if old_idx != new_idx and os.path.exists(old_path):
                            os.replace(old_path, new_path)
                        new_seeds.append(s)
                        new_idx += 1
                    seed_list = new_seeds
                    suc_num = len(seed_list)
                    epid = max(seed_list) + 1 if seed_list else 0
                    with open(os.path.join(args["save_path"], "seed.txt"), "w") as f:
                        for s in seed_list:
                            f.write("%s " % s)
                    print(f"[Backfill compact] {len(seed_list)} seeds remain; "
                          f"main loop will add {args['episode_num'] - len(seed_list)} replacements")

        while suc_num < args["episode_num"]:
            try:
                TASK_ENV.setup_demo(now_ep_num=suc_num, seed=epid, **args)
                TASK_ENV.play_once()

                if TASK_ENV.plan_success and TASK_ENV.check_success():
                    print(f"simulate data episode {suc_num} success! (seed = {epid})")
                    seed_list.append(epid)
                    TASK_ENV.save_traj_data(suc_num)
                    suc_num += 1
                else:
                    print(f"simulate data episode {suc_num} fail! (seed = {epid})")
                    fail_num += 1

                TASK_ENV.close_env()

                if args["render_freq"]:
                    TASK_ENV.viewer.close()
            except UnStableError as e:
                print(" -------------")
                print(f"simulate data episode {suc_num} fail! (seed = {epid})")
                print("Error: ", e)
                print(" -------------")
                fail_num += 1
                TASK_ENV.close_env()

                if args["render_freq"]:
                    TASK_ENV.viewer.close()
                time.sleep(0.3)
            except Exception as e:
                # stack_trace = traceback.format_exc()
                print(" -------------")
                print(f"simulate data episode {suc_num} fail! (seed = {epid})")
                print("Error: ", e)
                print(" -------------")
                fail_num += 1
                TASK_ENV.close_env()

                if args["render_freq"]:
                    TASK_ENV.viewer.close()
                time.sleep(1)

            epid += 1

            with open(os.path.join(args["save_path"], "seed.txt"), "w") as file:
                for sed in seed_list:
                    file.write("%s " % sed)

        print(f"\nComplete simulation, failed \033[91m{fail_num}\033[0m times / {epid} tries \n")
    else:
        print("\033[93m" + "Use Saved Seeds List".center(30, "-") + "\033[0m")
        with open(os.path.join(args["save_path"], "seed.txt"), "r") as file:
            seed_list = file.read().split()
            seed_list = [int(i) for i in seed_list]

    # =========== Collect Data ===========

    if args["collect_data"]:
        print("\033[93m" + "[Start Data Collection]" + "\033[0m")

        args["need_plan"] = False
        args["render_freq"] = 0
        args["save_data"] = True

        clear_cache_freq = args["clear_cache_freq"]

        st_idx = 0

        def exist_hdf5(idx):
            file_path = os.path.join(args["save_path"], 'data', f'episode{idx}.hdf5')
            return os.path.exists(file_path)

        while exist_hdf5(st_idx):
            st_idx += 1

        for episode_idx in range(st_idx, args["episode_num"]):
            if exist_hdf5(episode_idx):
                continue
            print(f"\033[34mTask name: {args['task_name']}\033[0m")

            TASK_ENV.setup_demo(now_ep_num=episode_idx, seed=seed_list[episode_idx], **args)

            traj_data = TASK_ENV.load_tran_data(episode_idx)
            args["left_joint_path"] = traj_data["left_joint_path"]
            args["right_joint_path"] = traj_data["right_joint_path"]
            TASK_ENV.set_path_lst(args)

            info_file_path = os.path.join(args["save_path"], "scene_info.json")

            if not os.path.exists(info_file_path):
                with open(info_file_path, "w", encoding="utf-8") as file:
                    json.dump({}, file, ensure_ascii=False)

            with open(info_file_path, "r", encoding="utf-8") as file:
                info_db = json.load(file)

            info = TASK_ENV.play_once()
            info_db[f"episode_{episode_idx}"] = info

            with open(info_file_path, "w", encoding="utf-8") as file:
                json.dump(info_db, file, ensure_ascii=False, indent=4)

            TASK_ENV.close_env(clear_cache=((episode_idx + 1) % clear_cache_freq == 0))
            TASK_ENV.merge_pkl_to_hdf5_video()
            TASK_ENV.remove_data_cache()
            assert TASK_ENV.check_success(), "Collect Error"

        instr_dir = os.path.join(args["save_path"], "instructions")
        all_instr_present = os.path.isdir(instr_dir) and all(
            os.path.exists(os.path.join(instr_dir, f"episode{i}.json"))
            for i in range(args["episode_num"])
        )
        if all_instr_present:
            print(f"\033[93m[Instructions] all {args['episode_num']} episode json files "
                  f"already present in {instr_dir} — skipping regeneration\033[0m")
        else:
            command = f"cd description && bash gen_episode_instructions.sh {args['task_name']} {args['task_config']} {args['language_num']}"
            os.system(command)


if __name__ == "__main__":
    from test_render import Sapien_TEST
    Sapien_TEST()

    import torch.multiprocessing as mp
    mp.set_start_method("spawn", force=True)

    parser = ArgumentParser()
    parser.add_argument("task_name", type=str)
    parser.add_argument("task_config", type=str)
    parser = parser.parse_args()
    task_name = parser.task_name
    task_config = parser.task_config

    main(task_name=task_name, task_config=task_config)
