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
const worldLandmarksPrint = document.getElementById("worldLandmarksPrint");
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
                console.log("result : ");
                console.log(result);
                // {
                //     "landmarks": [
                //         [
                //             {
                //                 "x": 0.4903971552848816,
                //                 "y": 0.17862549424171448,
                //                 "z": -0.5068303346633911
                //             },
                //             {
                //                 "x": 0.5036285519599915,
                //                 "y": 0.16429480910301208,
                //                 "z": -0.48697975277900696
                //             },...
                //         ]
                //     ],
                    // "worldLandmarks": [
                    //     [
                    //         {
                    //             "x": 0.01392222847789526,
                    //             "y": -0.6458527445793152,
                    //             "z": -0.309814453125
                    //         },
                    //         {
                    //             "x": 0.024383245036005974,
                    //             "y": -0.6807477474212646,
                    //             "z": -0.301513671875
                    //         },
                    //         ...
                    //     ]]
                    // worldLandmarksを抽出する
                    console.log("worldLandmarks : ");
                    console.log(result.worldLandmarks);
                // landmarksには、各部位の座標が入っている
// 0 - nose
// 1 - left eye (inner)
// 2 - left eye
// 3 - left eye (outer)
// 4 - right eye (inner)
// 5 - right eye
// 6 - right eye (outer)
// 7 - left ear
// 8 - right ear
// 9 - mouth (left)
// 10 - mouth (right)
// 11 - left shoulder
// 12 - right shoulder
// 13 - left elbow
// 14 - right elbow
// 15 - left wrist
// 16 - right wrist
// 17 - left pinky
// 18 - right pinky
// 19 - left index
// 20 - right index
// 21 - left thumb
// 22 - right thumb
// 23 - left hip
// 24 - right hip
// 25 - left knee
// 26 - right knee
// 27 - left ankle
// 28 - right ankle
// 29 - left heel
// 30 - right heel
// 31 - left foot index
// 32 - right foot index
                // worldLandmarksPrint.innerHTML = "33点の座標(単位：センチメートル)<br>X軸は腰の中心から右方向、Y軸は腰の中心から上方向、Z軸は腰の中心から前方向を示します。<br><br>";
                worldLandmarksPrint.innerHTML = "<h2>33点の座標(単位：センチメートル)</h2>";
                worldLandmarksPrint.innerHTML += "X軸は腰の中心から右方向、Y軸は腰の中心から上方向、Z軸は腰の中心から前方向を示します。<br><br>";
                worldLandmarksPrint.innerHTML += "点の読み取りについては、Google Mediapipeの公式ドキュメントを参照してください : <a href='https://ai.google.dev/edge/mediapipe/solutions/vision/pose_landmarker?hl=ja' target='_blank'>https://ai.google.dev/edge/mediapipe/solutions/vision/pose_landmarker?hl=ja</a><br><br>";
                const positionNamesJP = [
                    "鼻 (nose)",
                    "左目-内側 (left eye - inner)",
                    "左目 (left eye)",
                    "左目-外側 (left eye - outer)",
                    "右目-内側 (right eye - inner)",
                    "右目 (right eye)",
                    "右目-外側 (right eye - outer)",
                    "左耳 (left ear)",
                    "右耳 (right ear)",
                    "口-左縁 (mouth - left)",
                    "口-右縁 (mouth - right)",
                    "左肩 (left shoulder)",
                    "右肩 (right shoulder)",
                    "左肘 (left elbow)",
                    "右肘 (right elbow)",
                    "左手首 (left wrist)",
                    "右手首 (right wrist)",
                    "左小指 (left pinky)",
                    "右小指 (right pinky)",
                    "左人差し指 (left index)",
                    "右人差し指 (right index)",
                    "左親指 (left thumb)",
                    "右親指 (right thumb)",
                    "左腰 (left hip)",
                    "右腰 (right hip)",
                    "左膝 (left knee)",
                    "右膝 (right knee)",
                    "左足首 (left ankle)",
                    "右足首 (right ankle)",
                    "左かかと (left heel)",
                    "右かかと (right heel)",
                    "左足先 (left foot index)",
                    "右足先 (right foot index)"
                ];
                // 座標はx, y, zの3つの値で、それぞれ小数点以下3桁まで表示
                for (const [i, point] of result.worldLandmarks[0].entries()) {
                    worldLandmarksPrint.innerHTML += `${positionNamesJP[i]} : x = ${Math.round(point.x * 1000) / 10}cm, y = ${Math.round(point.y * 1000) / 10}cm, z = ${Math.round(point.z * 1000) / 10}cm<br>`;
                }
                worldLandmarksPrint.innerHTML += "<br>";
                // worldLandmarksPrint.innerHTML += "解析(1) : 腰(尻)の左右の座標と腰の中点の座標を求める<br>";
                worldLandmarksPrint.innerHTML += "<h2>解析(1) : 腰(尻)の左右の座標と腰の中点の座標を求める</h2>";
                const leftHip = result.worldLandmarks[0][23];
                const rightHip = result.worldLandmarks[0][24];
                const hipCenter = {
                    x: (leftHip.x + rightHip.x) / 2,
                    y: (leftHip.y + rightHip.y) / 2,
                    z: (leftHip.z + rightHip.z) / 2
                };
                worldLandmarksPrint.innerHTML += `左腰 : x = ${Math.round(leftHip.x * 1000) / 10}cm, y = ${Math.round(leftHip.y * 1000) / 10}cm, z = ${Math.round(leftHip.z * 1000) / 10}cm<br>`;
                worldLandmarksPrint.innerHTML += `右腰 : x = ${Math.round(rightHip.x * 1000) / 10}cm, y = ${Math.round(rightHip.y * 1000) / 10}cm, z = ${Math.round(rightHip.z * 1000) / 10}cm<br>`;
                worldLandmarksPrint.innerHTML += `腰の中点 : x = ${Math.round(hipCenter.x * 1000) / 10}cm, y = ${Math.round(hipCenter.y * 1000) / 10}cm, z = ${Math.round(hipCenter.z * 1000) / 10}cm<br><br>`;
                // worldLandmarksPrint.innerHTML += "解析(2) : 肩の左右の座標と肩の中点の座標を求める<br>";
                worldLandmarksPrint.innerHTML += "<h2>解析(2) : 肩の左右の座標と肩の中点の座標を求める</h2>";
                const leftShoulder = result.worldLandmarks[0][11];
                const rightShoulder = result.worldLandmarks[0][12];
                const shoulderCenter = {
                    x: (leftShoulder.x + rightShoulder.x) / 2,
                    y: (leftShoulder.y + rightShoulder.y) / 2,
                    z: (leftShoulder.z + rightShoulder.z) / 2
                };
                worldLandmarksPrint.innerHTML += `左肩 : x = ${Math.round(leftShoulder.x * 1000) / 10}cm, y = ${Math.round(leftShoulder.y * 1000) / 10}cm, z = ${Math.round(leftShoulder.z * 1000) / 10}cm<br>`;
                worldLandmarksPrint.innerHTML += `右肩 : x = ${Math.round(rightShoulder.x * 1000) / 10}cm, y = ${Math.round(rightShoulder.y * 1000) / 10}cm, z = ${Math.round(rightShoulder.z * 1000) / 10}cm<br>`;
                worldLandmarksPrint.innerHTML += `肩の中点 : x = ${Math.round(shoulderCenter.x * 1000) / 10}cm, y = ${Math.round(shoulderCenter.y * 1000) / 10}cm, z = ${Math.round(shoulderCenter.z * 1000) / 10}cm<br><br>`;
                // worldLandmarksPrint.innerHTML += "解析(3) : 左右腰を結ぶ線と左右肩を結ぶ線が(立体的に)どの程度平行かを求める<br>";
                worldLandmarksPrint.innerHTML += "<h2>解析(3) : 左右腰を結ぶ線と左右肩を結ぶ線が(立体的に)どの程度平行かを求める</h2>";
                const hipLine = {
                    x: rightHip.x - leftHip.x,
                    y: rightHip.y - leftHip.y,
                    z: rightHip.z - leftHip.z
                };
                const shoulderLine = {
                    x: rightShoulder.x - leftShoulder.x,
                    y: rightShoulder.y - leftShoulder.y,
                    z: rightShoulder.z - leftShoulder.z
                };
                const innerProduct = hipLine.x * shoulderLine.x + hipLine.y * shoulderLine.y + hipLine.z * shoulderLine.z;
                const hipLineNorm = Math.sqrt(hipLine.x ** 2 + hipLine.y ** 2 + hipLine.z ** 2);
                const shoulderLineNorm = Math.sqrt(shoulderLine.x ** 2 + shoulderLine.y ** 2 + shoulderLine.z ** 2);
                const cosTheta = innerProduct / (hipLineNorm * shoulderLineNorm);
                worldLandmarksPrint.innerHTML += `左右腰を結ぶ線 : x = ${Math.round(hipLine.x * 1000) / 10}cm, y = ${Math.round(hipLine.y * 1000) / 10}cm, z = ${Math.round(hipLine.z * 1000) / 10}cm<br>`;
                worldLandmarksPrint.innerHTML += `左右肩を結ぶ線 : x = ${Math.round(shoulderLine.x * 1000) / 10}cm, y = ${Math.round(shoulderLine.y * 1000) / 10}cm, z = ${Math.round(shoulderLine.z * 1000) / 10}cm<br>`;
                worldLandmarksPrint.innerHTML += `cos(左右腰と左右肩のなす角度) = ${Math.round(cosTheta * 1000) / 1000}<br><br>`;
                worldLandmarksPrint.innerHTML += `これは角度として${Math.round(Math.acos(cosTheta) * 180 / Math.PI * 1000) / 1000}度に相当します。<br><br>`;
                worldLandmarksPrint.innerHTML += "左右の肩を腰の線に対して平行になるように修正してこの角度を0にすることで、立ち姿勢を正すことができます。<br><br>";
                worldLandmarksPrint.innerHTML += "修正に必要な左の肩の移動量 (右の肩は固定したまま): <br>";
                worldLandmarksPrint.innerHTML += `x = ${Math.round((hipLine.x - shoulderLine.x) * 1000) / 10}cm, `;
                worldLandmarksPrint.innerHTML += `y = ${Math.round((hipLine.y - shoulderLine.y) * 1000) / 10}cm, `;
                worldLandmarksPrint.innerHTML += `z = ${Math.round((hipLine.z - shoulderLine.z) * 1000) / 10}cm<br><br>`;


                // worldLandmarksPrint.innerHTML += "解析(4) : 腰の中点と肩の中点を結ぶ線が(立体的に)どの程度垂直かを求める<br>";
                worldLandmarksPrint.innerHTML += "<h2>解析(4) : 腰の中点と肩の中点を結ぶ線が(立体的に)どの程度垂直かを求める</h2>";
                const hipShoulderLine = {
                    x: shoulderCenter.x - hipCenter.x,
                    y: shoulderCenter.y - hipCenter.y,
                    z: shoulderCenter.z - hipCenter.z
                };
                const innerProduct2 = hipLine.x * hipShoulderLine.x + hipLine.y * hipShoulderLine.y + hipLine.z * hipShoulderLine.z;
                const hipShoulderLineNorm = Math.sqrt(hipShoulderLine.x ** 2 + hipShoulderLine.y ** 2 + hipShoulderLine.z ** 2);
                const cosTheta2 = innerProduct2 / (hipLineNorm * hipShoulderLineNorm);
                worldLandmarksPrint.innerHTML += `腰の中点と肩の中点を結ぶ線 : x = ${Math.round(hipShoulderLine.x * 1000) / 10}cm, y = ${Math.round(hipShoulderLine.y * 1000) / 10}cm, z = ${Math.round(hipShoulderLine.z * 1000) / 10}cm<br>`;
                worldLandmarksPrint.innerHTML += `cos(左右腰と腰の中点と肩の中点を結ぶ線のなす角度) = ${Math.round(cosTheta2 * 1000) / 1000}<br><br>`;
                worldLandmarksPrint.innerHTML += `これは角度として${Math.round(Math.acos(cosTheta2) * 180 / Math.PI * 1000) / 1000}度に相当します。<br><br>`;
                worldLandmarksPrint.innerHTML += "腰の中点と肩の中点を結ぶ線を垂直にすることで、立ち姿勢のバランスを整えることができます。<br><br>";
                worldLandmarksPrint.innerHTML += "修正に必要な左右の肩の平行移動量(両方の肩を同じ方向に移動させる): <br>";
                // hipShoulderLineとhipLineの内積が0になるようなshoulderCenterの移動量を求める
                worldLandmarksPrint.innerHTML += `x = ${Math.round((hipLine.x * hipShoulderLine.x / hipLineNorm) * 1000) / 10}cm, `;
                worldLandmarksPrint.innerHTML += `y = ${Math.round((hipLine.y * hipShoulderLine.y / hipLineNorm) * 1000) / 10}cm, `;
                worldLandmarksPrint.innerHTML += `z = ${Math.round((hipLine.z * hipShoulderLine.z / hipLineNorm) * 1000) / 10}cm<br><br>`;

                worldLandmarksPrint.innerHTML += "<h2>結論</h2>";
                worldLandmarksPrint.innerHTML += "立ち姿勢を正すためには、<br>";
                worldLandmarksPrint.innerHTML += "右肩の移動量 : <br>";
                worldLandmarksPrint.innerHTML += `右(X)方向に ${Math.round((hipLine.x * hipShoulderLine.x / hipLineNorm) * 1000) / 10}cm、`;
                worldLandmarksPrint.innerHTML += `上(Y)方向に ${Math.round((hipLine.y * hipShoulderLine.y / hipLineNorm) * 1000) / 10}cm、`;
                worldLandmarksPrint.innerHTML += `前(Z)方向に ${Math.round((hipLine.z * hipShoulderLine.z / hipLineNorm) * 1000) / 10}cm<br>`;
                worldLandmarksPrint.innerHTML += "左肩の移動量 : <br>";
                worldLandmarksPrint.innerHTML += `右(X)方向に ${Math.round(((hipLine.x - shoulderLine.x) + (hipLine.x * hipShoulderLine.x / hipLineNorm)) * 1000) / 10}cm、`;
                worldLandmarksPrint.innerHTML += `上(Y)方向に ${Math.round(((hipLine.y - shoulderLine.y) + (hipLine.y * hipShoulderLine.y / hipLineNorm)) * 1000) / 10}cm、`;
                worldLandmarksPrint.innerHTML += `前(Z)方向に ${Math.round(((hipLine.z - shoulderLine.z) + (hipLine.z * hipShoulderLine.z / hipLineNorm)) * 1000) / 10}cm<br>`;





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
