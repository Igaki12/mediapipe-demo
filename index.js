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
        // segmentationMaskを有効にする
        // poseLandmarker.setOptions({ outputSegmentationMasks: true });
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
            }
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
            // 座標はx, y, zの3つの値で、それぞれ小数点以下3桁までテーブルで追加する
            //     <table id="worldLandmarksTable">
            //   <tr><th>部位</th><th>X座標/左方向(cm)</th><th>Y座標/下方向(cm)</th><th>Z座標/正面前方向(cm)</th></tr>
            //   </table>
            const worldLandmarksTable = document.getElementById("worldLandmarksTable");
            for (const [i, point] of result.worldLandmarks[0].entries()) {
                worldLandmarksTable.innerHTML += `<tr><td>${i + 1}. ${positionNamesJP[i]}</td><td>${Math.round(point.x * 1000) / 10}</td><td>${Math.round(point.y * 1000) / 10}</td><td>${Math.round(point.z * 1000) / 10}</td></tr>`;
            }
            // worldLandmarksTableの各列について、行の並び替え機能を追加する
            // ヘッダー行をクリックすると、その列を基準に行単位で並び替える
            let currentSortColumn = -1;
            let sortOrder = 1; // 1: 昇順, -1: 降順
            const tableHeaders = worldLandmarksTable.querySelectorAll("th");
            for (const [i, header] of tableHeaders.entries()) {
                header.addEventListener("click", () => {
                    // ヘッダー行のclassListをリセットする
                    tableHeaders.forEach((header) => {
                        header.classList.remove("sort-asc", "sort-desc");
                    });
                    if (currentSortColumn === i) {
                        sortOrder *= -1;
                    }
                    currentSortColumn = i;
                    // 並び替えを行う
                    header.classList.add(sortOrder === 1 ? "sort-asc" : "sort-desc");
                    const rows = Array.from(worldLandmarksTable.querySelectorAll("tr")).slice(1);
                    rows.sort((a, b) => {
                        // １列目以外は数値として比較する
                        if (i !== 0) {
                            const aValue = parseFloat(a.querySelectorAll("td")[i].textContent);
                            const bValue = parseFloat(b.querySelectorAll("td")[i].textContent);
                            return sortOrder * (aValue - bValue);
                        }
                        const aValue = a.querySelectorAll("td")[i].textContent;
                        const bValue = b.querySelectorAll("td")[i].textContent;
                        return sortOrder * aValue.localeCompare(bValue, undefined, { numeric: true });
                    });
                    for (const row of rows) {
                        worldLandmarksTable.appendChild(row);
                    }
                });
            }

            // 以下の解析を行う
            worldLandmarksPrint.innerHTML = "";
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
            worldLandmarksPrint.innerHTML += "修正に必要な左右の肩の移動量(左右肩の中点と肩の長さを保ったまま、左右の肩を中点に対して回転させる): <br>";
            const shoulderLength = Math.sqrt(shoulderLine.x ** 2 + shoulderLine.y ** 2 + shoulderLine.z ** 2);
            const hipLength = Math.sqrt(hipLine.x ** 2 + hipLine.y ** 2 + hipLine.z ** 2);
            const rightShoulderAfterAnalysis1 = {
                x: shoulderCenter.x + (0.5 * hipLine.x * shoulderLength / hipLength),
                y: shoulderCenter.y + (0.5 * hipLine.y * shoulderLength / hipLength),
                z: shoulderCenter.z + (0.5 * hipLine.z * shoulderLength / hipLength)
            };
            const leftShoulderAfterAnalysis1 = {
                x: shoulderCenter.x - (0.5 * hipLine.x * shoulderLength / hipLength),
                y: shoulderCenter.y - (0.5 * hipLine.y * shoulderLength / hipLength),
                z: shoulderCenter.z - (0.5 * hipLine.z * shoulderLength / hipLength)
            };
            worldLandmarksPrint.innerHTML += `右肩 : Δx = ${Math.round((rightShoulderAfterAnalysis1.x - rightShoulder.x) * 1000) / 10}cm, Δy = ${Math.round((rightShoulderAfterAnalysis1.y - rightShoulder.y) * 1000) / 10}cm, Δz = ${Math.round((rightShoulderAfterAnalysis1.z - rightShoulder.z) * 1000) / 10}cm<br>`;
            worldLandmarksPrint.innerHTML += `左肩 : Δx = ${Math.round((leftShoulderAfterAnalysis1.x - leftShoulder.x) * 1000) / 10}cm, Δy = ${Math.round((leftShoulderAfterAnalysis1.y - leftShoulder.y) * 1000) / 10}cm, Δz = ${Math.round((leftShoulderAfterAnalysis1.z - leftShoulder.z) * 1000) / 10}cm<br><br>`;

            // 同じ計算をlandmarksに対して行う
            const rightShoulderLandmark = result.landmarks[0][12];
            const leftShoulderLandmark = result.landmarks[0][11];
            const rightHipLandmark = result.landmarks[0][24];
            const leftHipLandmark = result.landmarks[0][23];
            const shoulderLineLandmark = {
                x: rightShoulderLandmark.x - leftShoulderLandmark.x,
                y: rightShoulderLandmark.y - leftShoulderLandmark.y,
                z: rightShoulderLandmark.z - leftShoulderLandmark.z
            };
            const shoulderLengthLandmark = Math.sqrt(shoulderLineLandmark.x ** 2 + shoulderLineLandmark.y ** 2 + shoulderLineLandmark.z ** 2);
            const hipLineLandmark = {
                x: rightHipLandmark.x - leftHipLandmark.x,
                y: rightHipLandmark.y - leftHipLandmark.y,
                z: rightHipLandmark.z - leftHipLandmark.z
            };
            const hipLengthLandmark = Math.sqrt(hipLineLandmark.x ** 2 + hipLineLandmark.y ** 2 + hipLineLandmark.z ** 2);
            const shoulderCenterLandmark = {
                x: (leftShoulderLandmark.x + rightShoulderLandmark.x) / 2,
                y: (leftShoulderLandmark.y + rightShoulderLandmark.y) / 2,
                z: (leftShoulderLandmark.z + rightShoulderLandmark.z) / 2
            };
            const rightShoulderAfterAnalysis1Landmark = {
                x: shoulderCenterLandmark.x + (0.5 * hipLineLandmark.x * shoulderLengthLandmark / hipLengthLandmark),
                y: shoulderCenterLandmark.y + (0.5 * hipLineLandmark.y * shoulderLengthLandmark / hipLengthLandmark),
                z: shoulderCenterLandmark.z + (0.5 * hipLineLandmark.z * shoulderLengthLandmark / hipLengthLandmark)
            };
            const leftShoulderAfterAnalysis1Landmark = {
                x: shoulderCenterLandmark.x - (0.5 * hipLineLandmark.x * shoulderLengthLandmark / hipLengthLandmark),
                y: shoulderCenterLandmark.y - (0.5 * hipLineLandmark.y * shoulderLengthLandmark / hipLengthLandmark),
                z: shoulderCenterLandmark.z - (0.5 * hipLineLandmark.z * shoulderLengthLandmark / hipLengthLandmark)
            };

// const drawingUtils = new DrawingUtils(canvasCtx);
// for (const landmark of result.landmarks) {
//     drawingUtils.drawLandmarks(landmark, {
//         radius: (data) => DrawingUtils.lerp(data.from?.z ?? 0, -0.15, 0.1, 5, 1)
//     });
            // drawingUtilsに、ただした後の左右肩の位置を描画する

            // drawingUtils.drawConnectors(landmark, PoseLandmarker.POSE_CONNECTIONS);

            worldLandmarksPrint.innerHTML += "<h2>解析(4) : 左右の肩を結ぶ線の中点が重心と一致するかを確認する</h2>";
            const leftAnkle = result.worldLandmarks[0][27];
            const rightAnkle = result.worldLandmarks[0][28];
            const ankleCenter = {
                x: (leftAnkle.x + rightAnkle.x) / 2,
                y: (leftAnkle.y + rightAnkle.y) / 2,
                z: (leftAnkle.z + rightAnkle.z) / 2
            };
            worldLandmarksPrint.innerHTML += `左右の足首を結んだ線の中点 : x = ${Math.round(ankleCenter.x * 1000) / 10}cm, y = ${Math.round(ankleCenter.y * 1000) / 10}cm, z = ${Math.round(ankleCenter.z * 1000) / 10}cm<br>`;
            worldLandmarksPrint.innerHTML += `左右の腰(尻)を結んだ線の中点 : x = ${Math.round(hipCenter.x * 1000) / 10}cm, y = ${Math.round(hipCenter.y * 1000) / 10}cm, z = ${Math.round(hipCenter.z * 1000) / 10}cm<br>`;
            worldLandmarksPrint.innerHTML += `左右の肩を結んだ線の中点 : x = ${Math.round(shoulderCenter.x * 1000) / 10}cm, y = ${Math.round(shoulderCenter.y * 1000) / 10}cm, z = ${Math.round(shoulderCenter.z * 1000) / 10}cm<br>`;
            worldLandmarksPrint.innerHTML += `この際に、背骨の長さ(肩の中点から腰の中点を結んだ線の長さ)は一定である　: `;
            const spineLength = Math.sqrt((shoulderCenter.x - hipCenter.x) ** 2 + (shoulderCenter.y - hipCenter.y) ** 2 + (shoulderCenter.z - hipCenter.z) ** 2);
            worldLandmarksPrint.innerHTML += `${Math.round(spineLength * 1000) / 10}cm<br><br>`;
            const hipAnkleLength = Math.sqrt((ankleCenter.x - hipCenter.x) ** 2 + (ankleCenter.y - hipCenter.y) ** 2 + (ankleCenter.z - hipCenter.z) ** 2);
            const hipAnkleLine = {
                x: hipCenter.x - ankleCenter.x,
                y: hipCenter.y - ankleCenter.y,
                z: hipCenter.z - ankleCenter.z
            };
            const shoulderCenterAfterAnalysis2 = {
                x: hipCenter.x + (hipAnkleLine.x * spineLength / hipAnkleLength),
                y: hipCenter.y + (hipAnkleLine.y * spineLength / hipAnkleLength),
                z: hipCenter.z + (hipAnkleLine.z * spineLength / hipAnkleLength)
            };
            worldLandmarksPrint.innerHTML += `解析4で導き出された肩の中点 : Δx = ${Math.round((shoulderCenterAfterAnalysis2.x - shoulderCenter.x) * 1000) / 10}cm, Δy = ${Math.round((shoulderCenterAfterAnalysis2.y - shoulderCenter.y) * 1000) / 10}cm, Δz = ${Math.round((shoulderCenterAfterAnalysis2.z - shoulderCenter.z) * 1000) / 10}cm<br><br>`;

            // 同じ計算をlandmarksに対して行う X,Y座標についてのみ行う
            const rightAnkleLandmark = result.landmarks[0][28];
            const leftAnkleLandmark = result.landmarks[0][27];
            const hipCenterLandmark = {
                x: (leftHipLandmark.x + rightHipLandmark.x) / 2,
                y: (leftHipLandmark.y + rightHipLandmark.y) / 2,
                z: (leftHipLandmark.z + rightHipLandmark.z) / 2
            };
            const ankleCenterLandmark = {
                x: (leftAnkleLandmark.x + rightAnkleLandmark.x) / 2,
                y: (leftAnkleLandmark.y + rightAnkleLandmark.y) / 2,
                z: (leftAnkleLandmark.z + rightAnkleLandmark.z) / 2
            };
            const hipAnkleLineLandmark = {
                x: hipCenterLandmark.x - ankleCenterLandmark.x,
                y: hipCenterLandmark.y - ankleCenterLandmark.y,
                z: hipCenterLandmark.z - ankleCenterLandmark.z
            };
            const hipAnkleLengthLandmarkXY = Math.sqrt(hipAnkleLineLandmark.x ** 2 + hipAnkleLineLandmark.y ** 2);
            const spineLengthLandmarkXY = Math.sqrt((shoulderCenterLandmark.x - hipCenterLandmark.x) ** 2 + (shoulderCenterLandmark.y - hipCenterLandmark.y) ** 2);
            const shoulderCenterAfterAnalysis2Landmark = {
                x: hipCenterLandmark.x + (hipAnkleLineLandmark.x * spineLengthLandmarkXY / hipAnkleLengthLandmarkXY),
                y: hipCenterLandmark.y + (hipAnkleLineLandmark.y * spineLengthLandmarkXY / hipAnkleLengthLandmarkXY),
                z: hipCenterLandmark.z + (hipAnkleLineLandmark.z * spineLengthLandmarkXY / hipAnkleLengthLandmarkXY)
            };

            worldLandmarksPrint.innerHTML += "<h2>結論</h2>";
            worldLandmarksPrint.innerHTML += "解析4で求めた肩の中点の移動量と、解析3で求めた両方の肩それぞれの移動量を合計することで、立ち姿勢を正すための肩の移動量を求めることができます。<br>";
            const rightShoulderAfterConclusion = {
                x: shoulderCenterAfterAnalysis2.x + (rightShoulderAfterAnalysis1.x - shoulderCenter.x),
                y: shoulderCenterAfterAnalysis2.y + (rightShoulderAfterAnalysis1.y - shoulderCenter.y),
                z: shoulderCenterAfterAnalysis2.z + (rightShoulderAfterAnalysis1.z - shoulderCenter.z)
            };
            const leftShoulderAfterConclusion = {
                x: shoulderCenterAfterAnalysis2.x + (leftShoulderAfterAnalysis1.x - shoulderCenter.x),
                y: shoulderCenterAfterAnalysis2.y + (leftShoulderAfterAnalysis1.y - shoulderCenter.y),
                z: shoulderCenterAfterAnalysis2.z + (leftShoulderAfterAnalysis1.z - shoulderCenter.z)
            };
            worldLandmarksPrint.innerHTML += `右肩 : Δx = ${Math.round((rightShoulderAfterConclusion.x - rightShoulder.x) * 1000) / 10}cm, Δy = ${Math.round((rightShoulderAfterConclusion.y - rightShoulder.y) * 1000) / 10}cm, Δz = ${Math.round((rightShoulderAfterConclusion.z - rightShoulder.z) * 1000) / 10}cm<br>`;
            worldLandmarksPrint.innerHTML += `左肩 : Δx = ${Math.round((leftShoulderAfterConclusion.x - leftShoulder.x) * 1000) / 10}cm, Δy = ${Math.round((leftShoulderAfterConclusion.y - leftShoulder.y) * 1000) / 10}cm, Δz = ${Math.round((leftShoulderAfterConclusion.z - leftShoulder.z) * 1000) / 10}cm<br><br>`;
            worldLandmarksPrint.innerHTML += "これらの移動量を元に、立ち姿勢を正すことができます。<br>";

            // 同じ計算をlandmarksに対して行う
            const rightShoulderAfterConclusionLandmark = {
                x: shoulderCenterAfterAnalysis2Landmark.x + (rightShoulderAfterAnalysis1Landmark.x - shoulderCenterLandmark.x),
                y: shoulderCenterAfterAnalysis2Landmark.y + (rightShoulderAfterAnalysis1Landmark.y - shoulderCenterLandmark.y),
                z: shoulderCenterAfterAnalysis2Landmark.z + (rightShoulderAfterAnalysis1Landmark.z - shoulderCenterLandmark.z)
            };
            const leftShoulderAfterConclusionLandmark = {
                x: shoulderCenterAfterAnalysis2Landmark.x + (leftShoulderAfterAnalysis1Landmark.x - shoulderCenterLandmark.x),
                y: shoulderCenterAfterAnalysis2Landmark.y + (leftShoulderAfterAnalysis1Landmark.y - shoulderCenterLandmark.y),
                z: shoulderCenterAfterAnalysis2Landmark.z + (leftShoulderAfterAnalysis1Landmark.z - shoulderCenterLandmark.z)
            };

            worldLandmarksPrint.innerHTML += `<br><br>画像に追加するための情報 : <br>右肩移動後 : x = ${Math.round(rightShoulderAfterConclusionLandmark.x * 1000) / 1000}, y = ${Math.round(rightShoulderAfterConclusionLandmark.y * 1000) / 1000}, z = ${Math.round(rightShoulderAfterConclusionLandmark.z * 1000) / 1000}<br>`;
            worldLandmarksPrint.innerHTML += `元の右肩座標 : x = ${Math.round(rightShoulderLandmark.x * 1000) / 1000}, y = ${Math.round(rightShoulderLandmark.y * 1000) / 1000}, z = ${Math.round(rightShoulderLandmark.z * 1000) / 1000}<br>`;
            worldLandmarksPrint.innerHTML += `左肩移動後 : x = ${Math.round(leftShoulderAfterConclusionLandmark.x * 1000) / 1000}, y = ${Math.round(leftShoulderAfterConclusionLandmark.y * 1000) / 1000}, z = ${Math.round(leftShoulderAfterConclusionLandmark.z * 1000) / 1000}<br>`;
            worldLandmarksPrint.innerHTML += `元の左肩座標 : x = ${Math.round(leftShoulderLandmark.x * 1000) / 1000}, y = ${Math.round(leftShoulderLandmark.y * 1000) / 1000}, z = ${Math.round(leftShoulderLandmark.z * 1000) / 1000}<br>`;
            worldLandmarksPrint.innerHTML += `肩の中点移動前 : x = ${Math.round(shoulderCenterLandmark.x * 1000) / 1000}, y = ${Math.round(shoulderCenterLandmark.y * 1000) / 1000}, z = ${Math.round(shoulderCenterLandmark.z * 1000) / 1000}<br>`;
            worldLandmarksPrint.innerHTML += `腰の中点移動前 : x = ${Math.round(hipCenterLandmark.x * 1000) / 1000}, y = ${Math.round(hipCenterLandmark.y * 1000) / 1000}, z = ${Math.round(hipCenterLandmark.z * 1000) / 1000}<br>`;
            worldLandmarksPrint.innerHTML += `spineLengthLandmark : ${Math.round(spineLengthLandmarkXY * 1000) / 1000}<br>`;
            worldLandmarksPrint.innerHTML += `hipAnkleLengthLandmark : ${Math.round(hipAnkleLengthLandmarkXY * 1000) / 1000}<br>`;
            // canvasに半径2の赤い点を描画する
// const canvas = document.createElement("canvas");
// canvas.setAttribute("class", "canvas");
// canvas.setAttribute("width", SelectedImage.naturalWidth + "px");
// canvas.setAttribute("height", SelectedImage.naturalHeight + "px");
// canvas.style =
//     "left: " + SelectedImage.offsetLeft + "px;" +
//     "top: " + SelectedImage.offsetTop + "px;" +
//     "width: " +
//     SelectedImage.width +
//     "px;" +
//     "height: " +
//     SelectedImage.height +
//     "px;";

// SelectedImage.parentNode.appendChild(canvas);

            const additionalCanvas = document.createElement("canvas");
            additionalCanvas.setAttribute("class", "canvas");
            additionalCanvas.setAttribute("width", SelectedImage.naturalWidth + "px");
            additionalCanvas.setAttribute("height", SelectedImage.naturalHeight + "px");
            additionalCanvas.style =
                "left: " + SelectedImage.offsetLeft + "px;" +
                "top: " + SelectedImage.offsetTop + "px;" +
                "width: " +
                SelectedImage.width +
                "px;" +
                "height: " +
                SelectedImage.height +
                "px;";
            SelectedImage.parentNode.appendChild(additionalCanvas);
            const additionalCanvasCtx = additionalCanvas.getContext("2d");
            const additionalDrawingUtils = new DrawingUtils(additionalCanvasCtx);
            additionalDrawingUtils.drawLandmarks([hipCenterLandmark,shoulderCenterLandmark,ankleCenterLandmark], {
                color: "orange",
                // radius: (data) => DrawingUtils.lerp(data.from?.z ?? 0, -0.15, 0.1, 5, 1)
                radius: 5
            });
            additionalDrawingUtils.drawLandmarks([rightShoulderLandmark,leftShoulderLandmark], {
                color: "orange",
                radius: 7
            });
            additionalDrawingUtils.drawLandmarks([shoulderCenterAfterAnalysis2Landmark,leftShoulderAfterConclusionLandmark,rightShoulderAfterConclusionLandmark], {
                color: "red",
                radius: 7
            });
            // 線の色を指定する
            additionalDrawingUtils.drawConnectors([shoulderCenterLandmark,hipCenterLandmark], PoseLandmarker.POSE_CONNECTIONS, {color: "orange"});
            // additionalDrawingUtils.drawConnectors([hipCenterLandmark,ankleCenterLandmark], PoseLandmarker.POSE_CONNECTIONS);

        }
        );
    }
}
);



// アップロードした動画を表示する
const selectedVideo = document.getElementById("selectedVideo");
const canvasVideo = document.getElementById("canvasVideo");
const videoResult = document.getElementById("videoResult");
const videoSelector = document.getElementById("videoSelector");
canvasVideo.style.left = selectedVideo.offsetLeft + "px";
canvasVideo.style.top = selectedVideo.offsetTop + "px";

videoSelector.addEventListener("change",async (event) => {
    const file = event.target.files[0];
    if (!file) {
        return;
    }
    const url = URL.createObjectURL(file);
    selectedVideo.src = url;
    selectedVideo.onload = () => {
        // if (!poseLandmarker) {
        //     console.log("Wait! poseLandmaker not loaded yet.");
        //     return;
        // }
        // if (runningMode === "IMAGE") {
        //     runningMode = "VIDEO";
        //     poseLandmarker.setOptions({ runningMode: "VIDEO" });
        // }
        // // returnがあるタイプのposeLandmarker.detectForVideoを使う: https://ai.google.dev/edge/api/mediapipe/js/tasks-vision.poselandmarker#poselandmarkerdetectforvideo
        // poseLandmarker.detectForVideo(selectedVideo, performance.now(), (result) => {
        //     videoResult.innerHTML = result;
        //     const canvasCtx = canvasVideo.getContext("2d");
        //     canvasCtx.save();
        //     canvasCtx.clearRect(0, 0, canvasVideo.width, canvasVideo.height);
        //     for (const landmark of result.landmarks) {
        //         drawingUtils.drawLandmarks(landmark, {
        //             radius: (data) => DrawingUtils.lerp(data.from?.z ?? 0, -0.15, 0.1, 5, 1)
        //         });
        //         drawingUtils.drawConnectors(landmark, PoseLandmarker.POSE_CONNECTIONS);
        //     }
        //     canvasCtx.restore();
        // }
        // );
        // return;

        
        // // 動画再生が開始されると、予測を開始する
        // selectedVideo.addEventListener("play", () => {
        //     console.log("Video started");
        //     predictVideo();
        // }
        // );

        // // 動画が再生されている限り、requestAnimationFrameを呼び出し続ける

        // async function predictVideo() {
        //     // 動画の再生が終了している場合、再生を停止する
        //     if (selectedVideo.ended) {
        //         console.log("Video ended");
        //         selectedVideo.pause();
        //         return;
        //     }
        //     // canvasに描画する
        //     canvas.style.height = selectedVideo.style.height;
        //     canvas.style.width = selectedVideo.style.width;
        //     // 予測を開始する
        //         poseLandmarker.detectForVideo(selectedVideo, performance.now(), (result) => {
        //             const canvasCtx = canvasVideo.getContext("2d");
        //             canvasCtx.save();
        //             canvasCtx.clearRect(0, 0, canvasVideo.width, canvasVideo.height);
        //             for (const landmark of result.landmarks) {
        //                 drawingUtils.drawLandmarks(landmark, {
        //                     radius: (data) => DrawingUtils.lerp(data.from?.z ?? 0, -0.15, 0.1, 5, 1)
        //                 });
        //                 drawingUtils.drawConnectors(landmark, PoseLandmarker.POSE_CONNECTIONS);
        //             }
        //             canvasCtx.restore();
        //         });
        //     if (!selectedVideo.paused) {
        //         window.requestAnimationFrame(predictVideo);
        //     }
        // }
    }
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
