// Copyright 2023 The MediaPipe Authors.

// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

import {
    PoseLandmarker,
    FilesetResolver,
    DrawingUtils
} from "https://cdn.skypack.dev/@mediapipe/tasks-vision@0.10.0";

const demosSection = document.getElementById("demos");

let poseLandmarker = undefined;
let runningMode = "IMAGE";
let enableWebcamButton;
let webcamRunning = false;
const videoHeight = "360px";
const videoWidth = "480px";

// Before we can use PoseLandmarker class we must wait for it to finish
// loading. Machine Learning models can be large and take a moment to
// get everything needed to run.
const createPoseLandmarker = async () => {
    const vision = await FilesetResolver.forVisionTasks(
        "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.0/wasm"
    );
    poseLandmarker = await PoseLandmarker.createFromOptions(vision, {
        baseOptions: {
            modelAssetPath: `https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task`,
            delegate: "GPU"
        },
        runningMode: runningMode,
        numPoses: 2
    });
    demosSection.classList.remove("invisible");
};
createPoseLandmarker();






/********************************************************************
// Demo 1: Grab a bunch of images from the page and detection them
// upon click.
********************************************************************/

// In this demo, we have put all our clickable images in divs with the
// CSS class 'detectionOnClick'. Lets get all the elements that have
// this class.
const imageContainers = document.getElementsByClassName("detectOnClick");

// Now let's go through all of these and add a click event listener.
for (let i = 0; i < imageContainers.length; i++) {
    // Add event listener to the child element whichis the img element.
    imageContainers[i].children[0].addEventListener("click", handleClick);
}

// When an image is clicked, let's detect it and display results!
async function handleClick(event) {
    if (!poseLandmarker) {
        console.log("Wait for poseLandmarker to load before clicking!");
        return;
    }

    if (runningMode === "VIDEO") {
        runningMode = "IMAGE";
        await poseLandmarker.setOptions({ runningMode: "IMAGE" });
    }
    // Remove all landmarks drawed before
    const allCanvas = event.target.parentNode.getElementsByClassName("canvas");
    for (var i = allCanvas.length - 1; i >= 0; i--) {
        const n = allCanvas[i];
        n.parentNode.removeChild(n);
    }

    // We can call poseLandmarker.detect as many times as we like with
    // different image data each time. The result is returned in a callback.
    poseLandmarker.detect(event.target, (result) => {
        const canvas = document.createElement("canvas");
        canvas.setAttribute("class", "canvas");
        canvas.setAttribute("width", event.target.naturalWidth + "px");
        canvas.setAttribute("height", event.target.naturalHeight + "px");
        canvas.style =
            "left: 0px;" +
            "top: 0px;" +
            "width: " +
            event.target.width +
            "px;" +
            "height: " +
            event.target.height +
            "px;";

        event.target.parentNode.appendChild(canvas);
        const canvasCtx = canvas.getContext("2d");
        const drawingUtils = new DrawingUtils(canvasCtx);
        for (const landmark of result.landmarks) {
            drawingUtils.drawLandmarks(landmark, {
                radius: (data) => DrawingUtils.lerp(data.from?.z ?? 0, -0.15, 0.1, 5, 1)
            });
            drawingUtils.drawConnectors(landmark, PoseLandmarker.POSE_CONNECTIONS);
        }
    });
}

// 手持ちの画像を選択して表示し、ポーズ推定を行う

const FileSelector = document.getElementById("fileSelector");
const SelectedImage = document.getElementById("selectedImage");
const landmarksPrint = document.getElementById("landmarksPrint");
FileSelector.addEventListener("change", (event) => {
    const file = event.target.files[0];
    if (!file) {
        return;
    }

    const reader = new FileReader();
    reader.onload = (e) => {
        SelectedImage.src = e.target.result;
    };
    reader.readAsDataURL(file);
    // 画像が表示されたら、ポーズ推定を行う
    SelectedImage.onload = () => {
        if (!poseLandmarker) {
            console.log("Wait for poseLandmarker to load before clicking!");
            return;
        }

        if (runningMode === "VIDEO") {
            runningMode = "IMAGE";
            poseLandmarker.setOptions({ runningMode: "IMAGE" });
        }
        const allCanvas = SelectedImage.parentNode.getElementsByClassName("canvas");
         poseLandmarker.detect(SelectedImage, async (result) => {
            const canvas = document.createElement("canvas");
            canvas.setAttribute("class", "canvas");
            canvas.setAttribute("width", SelectedImage.naturalWidth + "px");
            canvas.setAttribute("height", SelectedImage.naturalHeight + "px");
            canvas.style =
                "left: " + SelectedImage.offsetLeft + "px;" +
                "top: " + SelectedImage.offsetTop + "px;" +
                "width: " +
                SelectedImage.width +
                "px;" +
                "height: " +
                SelectedImage.height +
                "px;";

            SelectedImage.parentNode.appendChild(canvas);
            const canvasCtx = canvas.getContext("2d");
            const drawingUtils = new DrawingUtils(canvasCtx);
            for (const landmark of result.landmarks) {
                drawingUtils.drawLandmarks(landmark, {
                    radius: (data) => DrawingUtils.lerp(data.from?.z ?? 0, -0.15, 0.1, 5, 1)
                });
                drawingUtils.drawConnectors(landmark, PoseLandmarker.POSE_CONNECTIONS);

// [
//     {
//         "x": 0.4903971552848816,
//         "y": 0.17862549424171448,
//         "z": -0.5068303346633911
//     },
//     {
//         "x": 0.5036285519599915,
//         "y": 0.16429480910301208,
//         "z": -0.48697975277900696
//     },
//     {
//         "x": 0.512285590171814,
//         "y": 0.16482987999916077,
//         "z": -0.48729991912841797
//     },
//     {
//         "x": 0.5199361443519592,
//         "y": 0.16558152437210083,
//         "z": -0.487620085477829
//     },
//     {
//         "x": 0.47219109535217285,
//         "y": 0.1640891432762146,
//         "z": -0.48954111337661743
//     },
//     {
//         "x": 0.4605712592601776,
//         "y": 0.16464966535568237,
//         "z": -0.48986127972602844
//     },
//     {
//         "x": 0.4514756202697754,
//         "y": 0.1656760573387146,
//         "z": -0.48954111337661743
//     },
//     {
//         "x": 0.5273078680038452,
//         "y": 0.17907005548477173,
//         "z": -0.3403415381908417
//     },
//     {
//         "x": 0.44135335087776184,
//         "y": 0.1790199875831604,
//         "z": -0.35090717673301697
//     },
//     {
//         "x": 0.5068578720092773,
//         "y": 0.20093011856079102,
//         "z": -0.4488794207572937
//     },
//     {
//         "x": 0.4706171751022339,
//         "y": 0.20085135102272034,
//         "z": -0.4524013102054596
//     },
//     {
//         "x": 0.5916522145271301,
//         "y": 0.3003792464733124,
//         "z": -0.26061901450157166
//     },
//     {
//         "x": 0.3769046664237976,
//         "y": 0.3000282049179077,
//         "z": -0.2615795135498047
//     },
//     {
//         "x": 0.6073758602142334,
//         "y": 0.4385312795639038,
//         "z": -0.21083244681358337
//     },
//     {
//         "x": 0.35916170477867126,
//         "y": 0.43418699502944946,
//         "z": -0.20715048909187317
//     },
//     {
//         "x": 0.6119236946105957,
//         "y": 0.5572491884231567,
//         "z": -0.28175029158592224
//     },
//     {
//         "x": 0.3562454283237457,
//         "y": 0.5520846247673035,
//         "z": -0.2625400424003601
//     },
//     {
//         "x": 0.6168379783630371,
//         "y": 0.5945403575897217,
//         "z": -0.31360727548599243
//     },
//     {
//         "x": 0.35016119480133057,
//         "y": 0.5844627618789673,
//         "z": -0.29551762342453003
//     },
//     {
//         "x": 0.5999757051467896,
//         "y": 0.5923943519592285,
//         "z": -0.3560298979282379
//     },
//     {
//         "x": 0.36139512062072754,
//         "y": 0.5849749445915222,
//         "z": -0.33169692754745483
//     },
//     {
//         "x": 0.5968531370162964,
//         "y": 0.5822001099586487,
//         "z": -0.29983994364738464
//     },
//     {
//         "x": 0.36645767092704773,
//         "y": 0.5744699239730835,
//         "z": -0.27854856848716736
//     },
//     {
//         "x": 0.5380235910415649,
//         "y": 0.5457704663276672,
//         "z": -0.002264958107843995
//     },
//     {
//         "x": 0.41776043176651,
//         "y": 0.54132080078125,
//         "z": 0.0018910085782408714
//     },
//     {
//         "x": 0.524309515953064,
//         "y": 0.7046551704406738,
//         "z": 0.13567236065864563
//     },
//     {
//         "x": 0.42931458353996277,
//         "y": 0.702503502368927,
//         "z": 0.10717716813087463
//     },
//     {
//         "x": 0.5048248767852783,
//         "y": 0.8238471746444702,
//         "z": 0.5538954138755798
//     },
//     {
//         "x": 0.4473479390144348,
//         "y": 0.814140796661377,
//         "z": 0.5247598886489868
//     },
//     {
//         "x": 0.5002111792564392,
//         "y": 0.8289363384246826,
//         "z": 0.5907150506973267
//     },
//     {
//         "x": 0.4587453603744507,
//         "y": 0.8246767520904541,
//         "z": 0.561899721622467
//     },
//     {
//         "x": 0.5098884105682373,
//         "y": 0.8922178745269775,
//         "z": 0.430949866771698
//     },
//     {
//         "x": 0.4425351321697235,
//         "y": 0.8854601383209229,
//         "z": 0.3992529511451721
//     }
// ]
                // landmarksPrintに座標を表示
                console.log("landmark : ");
                console.log(landmark);
                landmarksPrint.innerHTML = "";
                for (const [i, point] of landmark.entries()) {
                    landmarksPrint.innerHTML += `landmark ${i} : x = ${point.x}, y = ${point.y}, z = ${point.z}<br>`;
                }
            }
        })};
        
}
);


/********************************************************************
// Demo 2: Continuously grab image from webcam stream and detect it.
********************************************************************/

const video = document.getElementById("webcam");
if (!(video instanceof HTMLVideoElement)) {
    throw new Error("Webcam element not found or is not a video element");
}
const canvasElement = document.getElementById("output_canvas");
if (!(canvasElement instanceof HTMLCanvasElement)) {
    throw new Error("Output canvas element not found or is not a canvas element");
}
const canvasCtx = canvasElement.getContext("2d");
const drawingUtils = new DrawingUtils(canvasCtx);

// Check if webcam access is supported.
const hasGetUserMedia = () => !!navigator.mediaDevices?.getUserMedia;

// If webcam supported, add event listener to button for when user
// wants to activate it.
if (hasGetUserMedia()) {
    enableWebcamButton = document.getElementById("webcamButton");
    enableWebcamButton.addEventListener("click", enableCam);
} else {
    console.warn("getUserMedia() is not supported by your browser");
}

// Enable the live webcam view and start detection.
function enableCam(event) {
    if (!poseLandmarker) {
        console.log("Wait! poseLandmaker not loaded yet.");
        return;
    }

    if (webcamRunning === true) {
        webcamRunning = false;
        enableWebcamButton.innerText = "ENABLE PREDICTIONS";
    } else {
        webcamRunning = true;
        enableWebcamButton.innerText = "DISABLE PREDICTIONS";
    }

    // getUsermedia parameters.
    const constraints = {
        video: true
    };

    // Activate the webcam stream.
    navigator.mediaDevices.getUserMedia(constraints).then((stream) => {
        video.srcObject = stream;
        video.addEventListener("loadeddata", predictWebcam);
    });
}

let lastVideoTime = -1;
async function predictWebcam() {
    canvasElement.style.height = videoHeight;
    video.style.height = videoHeight;
    canvasElement.style.width = videoWidth;
    video.style.width = videoWidth;
    // Now let's start detecting the stream.
    if (runningMode === "IMAGE") {
        runningMode = "VIDEO";
        await poseLandmarker.setOptions({ runningMode: "VIDEO" });
    }
    let startTimeMs = performance.now();
    if (lastVideoTime !== video.currentTime) {
        lastVideoTime = video.currentTime;
        poseLandmarker.detectForVideo(video, startTimeMs, (result) => {
            canvasCtx.save();
            canvasCtx.clearRect(0, 0, canvasElement.width, canvasElement.height);
            for (const landmark of result.landmarks) {
                drawingUtils.drawLandmarks(landmark, {
                    radius: (data) => DrawingUtils.lerp(data.from?.z ?? 0, -0.15, 0.1, 5, 1)
                });
                drawingUtils.drawConnectors(landmark, PoseLandmarker.POSE_CONNECTIONS);
            }
            canvasCtx.restore();
        });
    }

    // Call this function again to keep predicting when the browser is ready.
    if (webcamRunning === true) {
        window.requestAnimationFrame(predictWebcam);
    }
}
