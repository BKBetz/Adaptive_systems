## Opdracht 1.2

Agent.py contains the value iteration and the display of the maze.\
Policy.py contains the action function and the value function. \
Maze.py contains creating a maze, getting bordering states and making a step in the maze.\
\
Run main.py with the right settings.

1. Only want to see the agent walk, comment "show_utility" in agent.py line 57
2. Only want to see the utility of the maze after each iteration,
comment the agent.act() calls in main.py

Value iteration:\
Start:\
[[0. 0. 0.  0.]\
 [0. 0. 0. 0.]\
 [0. 0. 0. 0.]\
 [ 0. 0. 0. 0.]]\
Iteration 1:\
[[-1. -1. 40.  0.]\
 [-1. -1. -1. 40.]\
 [10. -1. -1. -1.]\
 [ 0. 10. -1. -1.]]\
Iteration 2:\
[[-2. 39. 40.  0.]\
 [ 9. -2. 39. 40.]\
 [10.  9. -2. 30.]\
 [ 0. 10.  8. -2.]]\
Iteration 3:\
[[38. 39. 40.  0.]\
 [ 9. 38. 39. 40.]\
 [10.  9. 29. 30.]\
 [ 0. 10.  8. 29.]]\
Iteration 4:\
[[38. 39. 40.  0.]\
 [37. 38. 39. 40.]\
 [10. 37. 29. 30.]\
 [ 0. 10. 28. 29.]]\
Iteration 5:\
[[38. 39. 40.  0.]\
 [37. 38. 39. 40.]\
 [36. 37. 36. 30.]\
 [ 0. 36. 28. 29.]]\
Iteration 6:\
[[38. 39. 40.  0.]\
 [37. 38. 39. 40.]\
 [36. 37. 36. 35.]\
 [ 0. 36. 35. 29.]]\
Iteration 7:\
[[38. 39. 40.  0.]\
 [37. 38. 39. 40.]\
 [36. 37. 36. 35.]\
 [ 0. 36. 35. 34.]]\
Iteration 8:\
[[38. 39. 40.  0.]\
 [37. 38. 39. 40.]\
 [36. 37. 36. 35.]\
 [ 0. 36. 35. 34.]]\