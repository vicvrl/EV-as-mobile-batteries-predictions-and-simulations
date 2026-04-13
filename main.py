import argparse
from simulation_runner import simulate_evs
from utils.initializers import clear_old_results
from config import EV_PATHS


def main():
    parser = argparse.ArgumentParser(description="Run smart and/or non-smart EV simulations.")
    parser.add_argument(
        "--mode",
        choices=["smart_gmm_i","smart_gmm_p","smart_lgbm","smart_sims","smart_2step","non_smart", "non_smart_no_public", "smart_oracle"],
        default="non_smart",
        help="Choose which simulation mode to run."
    )
    args = parser.parse_args()

    clear_old_results()
    simulate_evs(EV_PATHS, args.mode)


if __name__ == "__main__":
    main()


