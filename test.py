from coach_pl.tool.test import arg_parser, main

# Import this module to register the datasets, models, and modules.
import nerf

# Use the default testing script.
if __name__ == "__main__":
    args = arg_parser().parse_args()
    main(args)
