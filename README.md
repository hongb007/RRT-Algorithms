<!-- Improved compatibility of back to top link: See: https://github.com/othneildrew/Best-README-Template/pull/73 -->
<a id="readme-top"></a>
<!--
*** Thanks for checking out the Best-README-Template. If you have a suggestion
*** that would make this better, please fork the repo and create a pull request
*** or simply open an issue with the tag "enhancement".
*** Don't forget to give the project a star!
*** Thanks again! Now go create something AMAZING! :D
-->



<!-- PROJECT SHIELDS -->
<!--
*** I'm using markdown "reference style" links for readability.
*** Reference links are enclosed in brackets [ ] instead of parentheses ( ).
*** See the bottom of this document for the declaration of the reference variables
*** for contributors-url, forks-url, etc. This is an optional, concise syntax you may use.
*** https://www.markdownguide.org/basic-syntax/#reference-style-links
-->
[![Contributors][contributors-shield]][contributors-url]
[![Forks][forks-shield]][forks-url]
[![Stargazers][stars-shield]][stars-url]
[![Issues][issues-shield]][issues-url]
<!-- [![project_license][license-shield]][license-url]
[![LinkedIn][linkedin-shield]][linkedin-url] -->



<!-- PROJECT LOGO -->
<br />
<div align="center">
  <a href="https://github.com/github_username/repo_name">
    <img src="images/logo.png" alt="Logo" width="364" height="196">
  </a>
<h1 align="left">Rapidly-Exploring Random Tree (RRT) Algorithms</h1>
  <!-- <p align="center">
    <br />
    <a href="https://github.com/hongb007/RTT-Algorithms"><strong>Explore the docs »</strong></a>
    <br />
    <br />
    <a href="https://github.com/hongb007/RTT-Algorithms">View Demo</a>
    &middot;
    <a href="https://github.com/hongb007/RTT-Algorithms/issues/new?labels=bug&template=bug-report---.md">Report Bug</a>
    &middot;
    <a href="https://github.com/hongb007/RTT-Algorithms/issues/new?labels=enhancement&template=feature-request---.md">Request Feature</a>
  </p> -->
</div>



<!-- TABLE OF CONTENTS -->
<details>
  <summary>Table of Contents</summary>
  <ol>
    <li>
      <a href="#about-the-project">About The Project</a>
      <ul>
        <li><a href="#general-overview">General Overview</a></li>
        <li><a href="#rrt-parameter-optimization">RRT Parameter Optimization</a></li>
      </ul>
    </li>
    <li>
      <a href="#getting-started">Getting Started</a>
      <ul>
        <li><a href="#prerequisites">Prerequisites</a></li>
        <li><a href="#installation">Installation</a></li>
      </ul>
    </li>
    <li>
      <a href="#usage">Usage</a>
      <ul>
        <li><a href="#examples">Examples</a></li>
      </ul>
    </li>
    <!-- <li><a href="#roadmap">Roadmap</a></li> -->
    <li><a href="#contributing">Contributing</a></li>
    <!-- <li><a href="#license">License</a></li> -->
    <!-- <li><a href="#contact">Contact</a></li> -->
    <li><a href="#acknowledgments">Acknowledgments</a></li>
  </ol>
</details>



<!-- ABOUT THE PROJECT -->
## About The Project

### General Overview
<p>This project implements the Rapidly-Exploring Random Tree (RRT) algorithm for efficiently charting a path to goal in an environment with obstacles. RRT is applied frequently in high-dimensional spaces and is widely used in robotics for motion planning tasks. So far, this project features 2D planning around obstacles and an optimizer to find the optimal RRT parameters.</p>
<p>Key features of this implementation include:</p>
<ul>
  <li><strong>Obstacle Creation</strong>: Randomly generates rectangular obstacles defined within the search space.</li>
  <li><strong>Biased Node Steering</strong>: Connects a node to the tree in the direction of the goal a set percent of the time to more quickly find a valid path to the goal.</li>
  <li><strong>Angular Node Steering Constraint</strong>: Limits the angle at which a node can be randomly placed with respect to its corresponding parent node and the goal position a set percent of the time to speed up path finding.</li>
  <li><strong>Live Visual</strong>: Provides live plotting of the tree expansion and final path.</li>
  <li><strong>Parameter Optimization</strong>: Finds the set of parameters that will minimize the number samples to find a path to the goal in a given search space.</li>
</ul>

### RRT Parameter Optimization
This project implements Bayesian Optimization to find the optimal parameters for the constraints of the search space.
<details>
  <summary>More about the implementation</summary>
  <p>The optimization aims to minimize the number of samples required to find a path in a given search space. The parameters optimized include step_size, theta, turn_percent, and bias_percent. Furthermore, it features cumulative logging to persist optimization progress across multiple sessions, ensuring that the optimizer leverages previously gathered data.</p>
  <p>The optimization process is divided into multiple "sessions," where each session
  continues to refine the search based on the cumulative history of the past sessions. Each session consists of many RRT searches, or iterations, being executed under the same conditions (i.e. same obstacles). Upon completion, the optimizer outputs the overall best parameters found and also
  examines nearby good points to provide insights into the general optimal region, rather than just a single "lucky" parameter set.<p>

  To learn more about how the optimizer works, visit this [repository](https://github.com/bayesian-optimization/BayesianOptimization?tab=readme-ov-file).
  <h4>Excerpt from Baseyian optimization sessions (2 sessions, 10 iterations/session):</h4>

  ```
  --- Bayesian Optimization Session 2/2 ---
  Optimizer space has 25 points. Using 0 new initial random points, relying on history.
  Starting optimizer.maximize() with init_points=0 and n_iter=20.
  --- End of Session 2 ---
  Best parameters found by this optimizer instance (current overall best):
    Target: -117.00
    Parameters (rounded to 4 decimal places):
      - bias_percent: 15.2956
      - step_size: 5.4456
      - theta: 198.4974
      - turn_percent: 31.5858
  Total unique points known to this optimizer instance: 45

  --- Examining Optimization Results ---

  ### Fixed RRT Space Parameters
    Dimensions: [100 100]
    Start Position: [1 1]
    Goal Position: [99 99]
    Goal Radius: 3
    Max RRT Samples per iteration (n_samples): 1000
    Number of Rectangle Obstacles (n_rectangles): 75
    Rectangle Sizes( [[min_width, max_width] [min_height, max_height]]): [[ 5 15] [ 5 15]]
  ----------------------------------------

  ### Overall Best Point
    Target: -117.00
    Parameters (rounded to 4 decimal places):
      - bias_percent: 15.2956
      - step_size: 5.4456
      - theta: 198.4974
      - turn_percent: 31.5858

  ### Top 0 Nearby Good Points (within 30.00 of best)
    No other points found within the specified tolerance.

  Total time taken to run: 0h 8m 25.82s
  ```
</details>

<p align="right">(<a href="#readme-top">back to top</a>)</p>



<!-- GETTING STARTED -->
## Getting Started

To get a local copy up and running, follow the steps below.

### Prerequisites

[Python](https://www.python.org/downloads/) is installed and added to path.

### Installation

1. Clone the repository
   ```sh
   git clone https://github.com/hongb007/RTT-Algorithms.git
   ```
2. Install the packages
   ```sh
   pip install treelib numpy plotly matplotlib bayesian-optimization
   ```

<p align="right">(<a href="#readme-top">back to top</a>)</p>



<!-- USAGE EXAMPLES-->
## Usage
1. CD to the project directory

    ```sh
    cd .../RRT-Algorithms
    ```

2. Run the algorithm via terminal. To exit the program, close the plot.

    ```sh
    python rrt_2d.py
    ```

    If you don't want to plot the algorithm live or the end result, run

    ```sh 
    python rrt_2d.py --live False --plot_result False
    ```
3. Run optimizer on set paremeters

    ```sh
    python bayesian_optimization.py
    ```

    To change the number of sessions and iterations per session, change these values
    ```sh
    # ... (bayesian_optimization.py) ...

    n_bo_sessions = 5  # Number of Bayesian Optimization meta-iterations (sessions)
    n_bo_iterations_per_session = 50  # Adjust as needed (e.g., to 50-100 for real runs)
    ```

    After running the optimizer, copy and paste the optimized parameters from the terminal into rrt_2d.py and run the algorithm!
    ```sh
    # ... (rrt_2d.py) ...

    # Set the maximum distance the tree can extend in one iteration
    step_size = 7.04

    # Define the maximum turning angle in degrees
    theta = 180.0

    # Set the chance to turn a sample into the theta range from goal to parent node
    turn_percent = 65.0

    # Set the percentage bias towards sampling the goal directly
    bias_percent = 20.8
    ```
    ```sh 
    python rrt_2d.py --live True --plot_result True
    ```


### Examples
Plots from Baseyian optimized parameters:
<p>
  <img src="images/Example Plot 1.gif" alt="Example 1" width="400" height="300">
  <img src="images/Example Plot 2.gif" alt="Example 2" width="400" height="300">
</p>
<p align="right">(<a href="#readme-top">back to top</a>)</p>



<!-- ROADMAP -->
<!-- ## Roadmap

- [ ] Feature 1
- [ ] Feature 2
- [ ] Feature 3
    - [ ] Nested Feature

See the [open issues](https://github.com/hongb007/RTT-Algorithms/issues) for a full list of proposed features (and known issues).

<p align="right">(<a href="#readme-top">back to top</a>)</p> -->



<!-- CONTRIBUTING -->
## Contributing

Contributions are what make the open source community such an amazing place to learn, inspire, and create. Any contributions you make are **greatly appreciated**.

If you have a suggestion that would make this better, please fork the repo and create a pull request. You can also simply open an issue with the tag "enhancement".
Don't forget to give the project a star! Thanks again!

1. Fork the Project
2. Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
3. Commit your Changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the Branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

<p align="right">(<a href="#readme-top">back to top</a>)</p>

<!-- ### Top contributors:

<a href="https://github.com/hongb007/RTT-Algorithms/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=hongb007/RTT-Algorithms" alt="contrib.rocks image" />
</a> -->



<!-- LICENSE -->
<!-- ## License

Distributed under the project_license. See `LICENSE.txt` for more information.

<p align="right">(<a href="#readme-top">back to top</a>)</p> -->



<!-- CONTACT -->
<!-- ## Contact

Bill Hong - billhong@umich.edu

Project Link: [https://github.com/hongb007/RTT-Algorithms](https://github.com/hongb007/RTT-Algorithms)

<p align="right">(<a href="#readme-top">back to top</a>)</p> -->



<!-- ACKNOWLEDGMENTS -->
## Acknowledgments

* [Seth Issacson - sethgi@umich.edu](https://github.com/sethgi/)
<!-- * []()
* []() -->

<p align="right">(<a href="#readme-top">back to top</a>)</p>



<!-- MARKDOWN LINKS & IMAGES -->
<!-- https://www.markdownguide.org/basic-syntax/#reference-style-links -->
[contributors-shield]: https://img.shields.io/github/contributors/hongb007/RTT-Algorithms.svg?style=for-the-badge
[contributors-url]: https://github.com/hongb007/RTT-Algorithms/graphs/contributors
[forks-shield]: https://img.shields.io/github/forks/hongb007/RTT-Algorithms.svg?style=for-the-badge
[forks-url]: https://github.com/hongb007/RTT-Algorithms/network/members
[stars-shield]: https://img.shields.io/github/stars/hongb007/RTT-Algorithms.svg?style=for-the-badge
[stars-url]: https://github.com/hongb007/RTT-Algorithms/stargazers
[issues-shield]: https://img.shields.io/github/issues/hongb007/RTT-Algorithms.svg?style=for-the-badge
[issues-url]: https://github.com/hongb007/RTT-Algorithms/issues
[license-shield]: https://img.shields.io/github/license/hongb007/RTT-Algorithms.svg?style=for-the-badge
[license-url]: https://github.com/hongb007/RTT-Algorithms/blob/master/LICENSE.txt
[linkedin-shield]: https://img.shields.io/badge/-LinkedIn-black.svg?style=for-the-badge&logo=linkedin&colorB=555
[linkedin-url]: https://linkedin.com/in/linkedin_username
[product-screenshot]: images/screenshot.png
[Next.js]: https://img.shields.io/badge/next.js-000000?style=for-the-badge&logo=nextdotjs&logoColor=white
[Next-url]: https://nextjs.org/
[React.js]: https://img.shields.io/badge/React-20232A?style=for-the-badge&logo=react&logoColor=61DAFB
[React-url]: https://reactjs.org/
[Vue.js]: https://img.shields.io/badge/Vue.js-35495E?style=for-the-badge&logo=vuedotjs&logoColor=4FC08D
[Vue-url]: https://vuejs.org/
[Angular.io]: https://img.shields.io/badge/Angular-DD0031?style=for-the-badge&logo=angular&logoColor=white
[Angular-url]: https://angular.io/
[Svelte.dev]: https://img.shields.io/badge/Svelte-4A4A55?style=for-the-badge&logo=svelte&logoColor=FF3E00
[Svelte-url]: https://svelte.dev/
[Laravel.com]: https://img.shields.io/badge/Laravel-FF2D20?style=for-the-badge&logo=laravel&logoColor=white
[Laravel-url]: https://laravel.com
[Bootstrap.com]: https://img.shields.io/badge/Bootstrap-563D7C?style=for-the-badge&logo=bootstrap&logoColor=white
[Bootstrap-url]: https://getbootstrap.com
[JQuery.com]: https://img.shields.io/badge/jQuery-0769AD?style=for-the-badge&logo=jquery&logoColor=white
[JQuery-url]: https://jquery.com 