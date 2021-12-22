# TraceArmature Summary #

TraceArmature is currently a set of python scripts that allow for high fidelity motion capture through the use of AI pose estimation (using METRABS), fiducial markers, VR body trackers, and optional hand annotations. This information is applied to a rigged 3D scan of the subject and further refined using differentiable rendering and AKAZE features. Eventually, this project will be made into a blender plugin. The purpose of this project is to allow for UV mapping of an arbitrary target (human or non-human) with higher fidelity than is possible with AI pose estimation alone and with more accuracy and control than dense pose estimation. The depth and UV information gathered can be used to aid in style transfer techniques like Adobe's patch match algorithm or various neural style transfer methods. It is important to note that the aim of this project is accuracy, not speed. Running the highest quality UV reconstruction can take around 30 seconds per frame.

It is important to note that this project is still under HEAVY development, so formal instructions will be delayed until a stable, user-friendly build is ready.

![Wireframe Motion Tracking Demo](images/TraceArmatureGIF.gif)
