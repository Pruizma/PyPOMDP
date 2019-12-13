
## How to run it

usage: main.py config [--env ENV] [--max_play MAX_PLAY] [--benchmark BENCHMARK]

Required arguments:

  config                The file name of algorithm configuration(pomcp, pbvi)

Optional arguments:
  
  --env ENV                 The name of environment's config file (Tiger-2D.POMDP, Islands.POMDP, Tag.POMDP, Web.POMDP)
  --max_play MAX_PLAY       Maximum number of play steps (maximum steps)
  --benchmark BENCHMARK     Maximum number of benchmark simulations (simulations)

Example usages:
> python main.py pomcp --env Tiger-2D.POMDP
> python main.py pomcp --env Web.POMDP --max_play 4
> python main.py pbvi --env Landing.POMDP --benchmark 30

