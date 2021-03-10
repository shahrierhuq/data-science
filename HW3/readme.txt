Unfortunately I could not solve the argparse errors I was receiving, so I cannot run the program through the command line.

Run example: 

python3 main.py --param param/param.json -v 2 --res-path plots
--x-field "-y/np.sqrt(x**2 + y**2)" --y-field "x/np.sqrt(x**2 + y**2)"
--lb -1.0 --ub 1.0 --n-tests 3