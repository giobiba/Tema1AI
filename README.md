# Homework #1 AI
This project represents my first project that I had in the knowledge representation course. This homework consisted of applying different search algorithms to find the best solution of a given game and figuring out the pros and cons for each algorithm. In my game you had to remove clusters of blocks from a 2D plane until no more legal moves were available. A legal move meant that you remove the maximum amount of adjacent blocks of the same color, with a minimum of 3 blocks. After you move, the plain would be shifted down and left for all the non empty blocks.
The algorithms used in this homework are the following:
* Uniform cost search
* A star
* A star with open closed optimization
* Iterative deepening A star

What I observed while trying these various algorithms was that A* performed a bit better than IDA* (28 miliseconds compared to 45), but in terms of memory IDA* made a big difference, having 36 search nodes in memory (at maximum) while A* had up to 7 times more.

I also tried using four different heuristics for estimating the cost, one overshooting and resulting in non optimal solutions and one ordinary heuristic. The other two were created based on the cost equation that stated that the cost of a move was: <br>
`cost = 1 + (N - K)/N` where K is the number of eliminated blocks, and N the total number of blocks of the same color with the eliminated blocks. With that idea in mind, the heuristic that I first came up with estimated the cost to be the number of colors remaining. The second one was a simplified version of the first one.
