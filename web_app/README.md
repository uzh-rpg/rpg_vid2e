<div id="top"></div>
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
<!-- [![Contributors][contributors-shield]][contributors-url]
[![Forks][forks-shield]][forks-url]
[![Stargazers][stars-shield]][stars-url]
[![Issues][issues-shield]][issues-url]
[![MIT License][license-shield]][license-url]
[![LinkedIn][linkedin-shield]][linkedin-url] -->



<!-- PROJECT LOGO -->
<br />
<div align="center">
  <!-- <a href="https://github.com/othneildrew/Best-README-Template">
    <img src="images/logo.png" alt="Logo" width="80" height="80">
  </a> -->

  <h2 align="center">ESIM Web App Playground</h3>

  <p align="center">
    A Web App for Generating Events from Upsampled Video Frames & Visualization of Events from Live Video of a Webcam! 
    <!-- <br />
    <a href="https://github.com/othneildrew/Best-README-Template"><strong>Explore the docs »</strong></a>
    <br />
    <br />
    <a href="https://github.com/othneildrew/Best-README-Template">View Demo</a>
    ·
    <a href="https://github.com/othneildrew/Best-README-Template/issues">Report Bug</a>
    ·
    <a href="https://github.com/othneildrew/Best-README-Template/issues">Request Feature</a> -->
    <br />
    <br />
    <a href="https://github.com/iamsiddhantsahu">Siddhant Sahu</a> | <a href="https://github.com/danielgehrig18">Daniel Gehrig</a> | <a href="">Nico Messikommer</a>

  </p>
</div>

![Product Name Screen Shot](screencast.gif)

<!-- TABLE OF CONTENTS -->
<details>
  <summary>Table of Contents</summary>
  <ol>
    <li>
      <a href="#about-the-project">About The Project</a>
      <!-- <ul>
        <li><a href="#built-with">Built With</a></li>
      </ul> -->
    </li>
    <li>
      <a href="#getting-started">Getting Started</a>
      <ul>
        <li><a href="#prerequisites">Prerequisites & Installation</a></li>
        <li><a href="#Running the Web App Locally">Running the Web App Locally</a></li>
        <li><a href="#Running the Web App on Google Colab">Running the Web App on Google Colab</a></li>
      </ul>
    </li>
    <!-- <li><a href="#usage">Usage</a></li> -->
    <li><a href="#roadmap">Roadmap</a></li>
    <!-- <li><a href="#contributing">Contributing</a></li>
    <li><a href="#license">License</a></li>
    <li><a href="#contact">Contact</a></li>
    <li><a href="#acknowledgments">Acknowledgments</a></li> -->
  </ol>
</details>



<!-- ABOUT THE PROJECT -->
## About The Project

<!-- ![Product Name Screen Shot](screencast.gif) -->

The goal of this project is to make ESIM and VID2E available to researchers who do not possess a real event camera. For this we want to deploy VID2E and ESIM as an interactive web app. The app should be easy to use and have the following functional requirements:

* Generation of events from video through dragging and dropping a video into the browser. The resulting events should be downloadable as raw events and rendered video.
* Incorporate video interpolation via an existing video interpolation method (e.g. Super SloMo) before event generation with ESIM.
* Visualization and inspection of the event stream in the browser.
* The event generation should be configurable by changing ESIM parameters.

<p align="right">(<a href="#top">back to top</a>)</p>



<!-- ### Built With

This section should list any major frameworks/libraries used to bootstrap your project. Leave any add-ons/plugins for the acknowledgements section. Here are a few examples.

* [Next.js](https://nextjs.org/)
* [React.js](https://reactjs.org/)
* [Vue.js](https://vuejs.org/)
* [Angular](https://angular.io/)
* [Svelte](https://svelte.dev/)
* [Laravel](https://laravel.com)
* [Bootstrap](https://getbootstrap.com)
* [JQuery](https://jquery.com)

<p align="right">(<a href="#top">back to top</a>)</p> -->



<!-- GETTING STARTED -->
## Getting Started

Follow these steps in order to make this app run in your local system.

_This code base assumes you have access to a NVIDIA GPU in your system with proper drivers installed along with the `cuda-toolkit` version `10.1`._

### Prerequisites & Installation

Make sure that you have followed the [Instalation with Anaconda Instruction](https://github.com/uzh-rpg/rpg_vid2e) and have created the `vid2e` Conda environment with the `esim_torch` package installed in this environment.
1. Activate the Conda environment
  ```sh
  conda activate vid2e
  conda list | grep 'esim'
  ```

  The above command should output the following
  ```sh
  esim-cuda                 0.0.0                    pypi_0    pypi
  esim-py                   0.0.1                    pypi_0    pypi
  ```

2. Install the `streamlit` package in this Conda environment, along with a couple of additional packages.
```sh
pip install streamlit stqdm numba h5py
```
### Running the Web App Locally

If you want to run this app locally on your system then follow these steps.
1. Run the App
```sh
streamlit run web_app.py
```
2. Then you can access the app in Local URL: http://localhost:8501

_When using the 'video upload' feature, the uploaded file is saved in the 'data/original/video_upload' directory. Subsequently, upon clicking the 'Generate Events' button, two new directory, namely, 'data/upsampled/video_upload' and 'data/events/video_upload' are created. Additionally, after the process has finished, you will see a 'Download Events' button, you can click this button to download the events. Events can be downloaded in three different formats: as .h5 file, as .npz or as a rendered video._

<p align="right">(<a href="#top">back to top</a>)</p>

### Running the Web App on Google Colab

To run this web app on Google Colab follow these steps in order.

_Make sure you select the `Run time type` as `GPU` in `Google Colab`_

1. Register and create an account in the [remote.it](https://remote.it/).
2. Install remote.it service in your Google Colab instance.
  ```sh
  !curl -LkO https://raw.githubusercontent.com/remoteit/installer/master/scripts/auto-install.sh
  ! chmod +x ./auto-install.sh
  ! sudo ./auto-install.sh
  ```
3. Install Miniconda in your Google Colab Instance
  ```sh
  ! wget https://repo.anaconda.com/miniconda/Miniconda3-py37_4.8.2-Linux-x86_64.sh
  ! chmod +x Miniconda3-py37_4.8.2-Linux-x86_64.sh
  ! bash ./Miniconda3-py37_4.8.2-Linux-x86_64.sh -b -f -p /usr/local
  ```
4. Update Python Path
  ```sh
  import sys
  sys.path.insert(0,'/usr/local/lib/python3.7/site-packages/')
  ```
5. Install Streamlit package
  ```sh
  !pip install streamlit
  ```
6. Run the remote.it service
  ```sh
  !sudo connectd_installer
  ```
  - In the menu selection choose `1`
  - Enter your remote.it `email address` and `password`
  - Enter the name of your device `esim`
  - Choose `1` `Attach/reinstall a remote.it Service to an application`
  - Protocol Selection Menu, Choose `2`, `Web (HTTP) on port 80`
  - Enter a name for this remote.it service, `esim`
  - Main menu, choose, `5` `Exit`
7. Mount your Google Drive to this Google Colab Instance
8. Change directory to where to want this project to reside.
9. Clone the repo
   ```sh
   git clone https://github.com/uzh-rpg/rpg_vid2e.git
   ```
10. Check if the `esim_torch` package is installed
   ```sh
   import esim
   print(esim_torch.__path__)
   ```
11. Built the `esim_torch` package with `pybind11`
   ```sh
   pip install esim_torch/
   ```
12. Change directory inside `web_app` and run the app on port number `80` as a webservice which can then be accessed for the remote.it service.
   ```sh
   !streamlit run --server.port 80 web_app.py&>/dev/null&
   ```

Finaly, you can navigate your remote.it dashboard and open these service and use the app!

_Note: Remeber when running this app on Google Colab you need to change the paths for input and output directories for the upsampling to the specific directory in Google Drive and also other paths and esim import._

<p align="right">(<a href="#top">back to top</a>)</p>


<!-- ROADMAP -->
### Roadmap

- [x] Add video upload dragging and dropping a video into the browser.
- [x] Add video interpolation via an existing video interpolation method (e.g. Super SloMo) before event generation with ESIM.
- [x] Add events download button (output format)
    - [x] HDF5
    - [x] NPZ
    - [x] Rendered Video
- [x] Add webcam functionality
- [x] Run this app on Google Colab



<!-- CONTRIBUTING
## Contributing

Contributions are what make the open source community such an amazing place to learn, inspire, and create. Any contributions you make are **greatly appreciated**.

If you have a suggestion that would make this better, please fork the repo and create a pull request. You can also simply open an issue with the tag "enhancement".
Don't forget to give the project a star! Thanks again!

1. Fork the Project
2. Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
3. Commit your Changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the Branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

<p align="right">(<a href="#top">back to top</a>)</p> -->



<!-- LICENSE
## License

Distributed under the MIT License. See `LICENSE.txt` for more information.

<p align="right">(<a href="#top">back to top</a>)</p> -->



<!-- CONTACT
## Contact

Siddhant Sahu
Daniel Gehrig
Nico Messikommer

<!-- Siddhant Sahu - [@your_twitter](https://twitter.com/your_username) - email@example.com -->

<!-- Project Link: [https://github.com/your_username/repo_name](https://github.com/your_username/repo_name) -->

<!-- <p align="right">(<a href="#top">back to top</a>)</p> -->



<!-- ACKNOWLEDGMENTS
## Acknowledgments

Use this space to list resources you find helpful and would like to give credit to. I've included a few of my favorites to kick things off!

* [Choose an Open Source License](https://choosealicense.com)
* [GitHub Emoji Cheat Sheet](https://www.webpagefx.com/tools/emoji-cheat-sheet)
* [Malven's Flexbox Cheatsheet](https://flexbox.malven.co/)
* [Malven's Grid Cheatsheet](https://grid.malven.co/)
* [Img Shields](https://shields.io)
* [GitHub Pages](https://pages.github.com)
* [Font Awesome](https://fontawesome.com)
* [React Icons](https://react-icons.github.io/react-icons/search)

<p align="right">(<a href="#top">back to top</a>)</p> -->



<!-- MARKDOWN LINKS & IMAGES -->
<!-- https://www.markdownguide.org/basic-syntax/#reference-style-links -->
[contributors-shield]: https://img.shields.io/github/contributors/othneildrew/Best-README-Template.svg?style=for-the-badge
[contributors-url]: https://github.com/othneildrew/Best-README-Template/graphs/contributors
[forks-shield]: https://img.shields.io/github/forks/othneildrew/Best-README-Template.svg?style=for-the-badge
[forks-url]: https://github.com/othneildrew/Best-README-Template/network/members
[stars-shield]: https://img.shields.io/github/stars/othneildrew/Best-README-Template.svg?style=for-the-badge
[stars-url]: https://github.com/othneildrew/Best-README-Template/stargazers
[issues-shield]: https://img.shields.io/github/issues/othneildrew/Best-README-Template.svg?style=for-the-badge
[issues-url]: https://github.com/othneildrew/Best-README-Template/issues
[license-shield]: https://img.shields.io/github/license/othneildrew/Best-README-Template.svg?style=for-the-badge
[license-url]: https://github.com/othneildrew/Best-README-Template/blob/master/LICENSE.txt
[linkedin-shield]: https://img.shields.io/badge/-LinkedIn-black.svg?style=for-the-badge&logo=linkedin&colorB=555
[linkedin-url]: https://linkedin.com/in/othneildrew
[product-screenshot]: images/screenshot.png
