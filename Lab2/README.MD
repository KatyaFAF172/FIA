# Laboratory work Nr.1

Task
===

Find an example implementation of the A* algorithm and execute it on your computer. Answer
the following questions:
1. What is A*?
2. What are 3 examples where A* could be used?
3. What are some alternatives to A*?
4. What is a heuristic?
5. What is a boid?

**1. What is A*?**

What is A*? The A* search algorithm is useful for nding the lowest cost path between
two nodes (aka vertices) of a graph. The path may traverse any number of nodes connected
by edges (aka arcs) with each edge having an associated cost. Each time A* enters a
state, it calculates the cost, f(n) (n being the neighboring node), to travel to all of the
neighboring nodes, and then enters the node with the lowest value of f(n). These values
are calculated with the following formula:

f(n) = g(n) + h(n)

g(n) being the value of the shortest path from the start node to node n, and h(n) being
a heuristic approximation of the node's value.
For us to be able to reconstruct any path, we need to mark every node with the relative
that has the optimal f(n) value. This also means that if we revisit certain nodes, we'll
have to update their most optimal relatives as well. More on that later.

**2.What are 3 examples where A* could be used?**

Search Space
In any game environment, AI characters need to use an underlying data structure { a
search space representation { to plan a path to any given destination. Finding the most
appropriate data structure to represent the search space for the game world is absolutely
critical to achieving realistic-looking movement and acceptable pathnding performance.

In maps the A* algorithm is used to calculate the shortest distance between the source
(initial state) and the destination (nal state).

It is used in distance calculations, in mazes, in pathding from point A to point B.

**3. What are some alternatives to A*?**

Dijkstra's Algorithm works by visiting vertices in the graph starting with the object's
starting point. It then repeatedly examines the closest not-yet-examined vertex, adding
its vertices to the set of vertices to be examined. It expands outwards from the starting
point until it reaches the goal. Dijkstra's Algorithm is guaranteed to nd a shortest path
from the starting point to the goal, as long as none of the edges have a negative cost.
The Greedy Best-First-Search algorithm works in a similar way, except that it has some
estimate (called a heuristic) of how far from the goal any vertex is. Instead of selecting
the vertex closest to the starting point, it selects the vertex closest to the goal. Greedy
Best-First-Search is not guaranteed to nd a shortest path. However, it runs much quicker
than Dijkstra's Algorithm because it uses the heuristic function to guide its way towards
the goal very quickly. For example, if the goal is to the south of the starting position,
Greedy Best-First-Search will tend to focus on paths that lead southwards.

**4. What is a heuristic?**

Heuristic is a function which is used in Informed Search, and it nds the most promising
path. It takes the current state of the agent as its input and produces the estimation of
how close agent is from the goal.

**5. What is a boid?**

Boids is an articial life simulation originally developed by Craig Reynolds. Is generic
simulated 
ocking creature. The aim of the simulation was to replicate the behavior of

ocks of birds and sh.
How does it work? Each of the boids (bird-oid objects) obeys three simple rules:
1. Coherence Each boid 
ies towards the the other boids. But they don't just immediately

y directly at each other.
2. Separation Each boid also tries to avoid running into the other boids. If it gets too
close to another boid it will steer away from it.
3. Alignment Finally, each boid tries to match the vector (speed and direction) of the
other boids around it.
