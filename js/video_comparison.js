// Written by Dor Verbin, October 2021
// This is based on: http://thenewcode.com/364/Interactive-Before-and-After-Video-Comparison-in-HTML5-Canvas
// With additional modifications based on: https://jsfiddle.net/7sk5k4gp/13/

function playVids(videoId) {
    var videoMerge = document.getElementById(videoId + "Merge");
    var vid = document.getElementById(videoId);

    var position = 0.5;
    var vidWidth = vid.videoWidth/2;
    var vidHeight = vid.videoHeight;

    var mergeContext = videoMerge.getContext("2d");

    
    if (vid.readyState > 3) {
        vid.play();

        function trackLocation(e) {
            // Normalize to [0, 1]
            bcr = videoMerge.getBoundingClientRect();
            position = ((e.pageX - bcr.x) / bcr.width);
        }
        function trackLocationTouch(e) {
            // Normalize to [0, 1]
            bcr = videoMerge.getBoundingClientRect();
            position = ((e.touches[0].pageX - bcr.x) / bcr.width);
        }

        videoMerge.addEventListener("mousemove",  trackLocation, false); 
        videoMerge.addEventListener("touchstart", trackLocationTouch, false);
        videoMerge.addEventListener("touchmove",  trackLocationTouch, false);


        function drawLoop() {
            mergeContext.drawImage(vid, 0, 0, vidWidth, vidHeight, 0, 0, vidWidth, vidHeight);
            var colStart = (vidWidth * position).clamp(0.0, vidWidth);
            var colWidth = (vidWidth - (vidWidth * position)).clamp(0.0, vidWidth);
            mergeContext.drawImage(vid, colStart+vidWidth, 0, colWidth, vidHeight, colStart, 0, colWidth, vidHeight);
            requestAnimationFrame(drawLoop);

            
            var arrowLength = 0.09 * vidHeight;
            var arrowheadWidth = 0.025 * vidHeight;
            var arrowheadLength = 0.04 * vidHeight;
            var arrowPosY = vidHeight / 2;
            var arrowWidth = 0.007 * vidHeight;
            var currX = vidWidth * position;

            // Draw circle
            mergeContext.arc(currX, arrowPosY, arrowLength*0.7, 0, Math.PI * 2, false);
            mergeContext.fillStyle = "#FFFFFF";
            mergeContext.fill()
            //mergeContext.strokeStyle = "#444444";
            //mergeContext.stroke()
            
            // Draw border
            mergeContext.beginPath();
            mergeContext.moveTo(vidWidth*position, 0);
            mergeContext.lineTo(vidWidth*position, vidHeight);
            mergeContext.closePath()
            mergeContext.strokeStyle = "#FFFFFF";
            mergeContext.lineWidth = 5;            
            mergeContext.stroke();

            // Draw arrow
            mergeContext.beginPath();
            mergeContext.moveTo(currX, arrowPosY - arrowWidth/2);
            
            // Move right until meeting arrow head
            mergeContext.lineTo(currX + arrowLength/2 - arrowheadLength/2, arrowPosY - arrowWidth/2);
            
            // Draw right arrow head
            mergeContext.lineTo(currX + arrowLength/2 - arrowheadLength/2, arrowPosY - arrowheadWidth/2);
            mergeContext.lineTo(currX + arrowLength/2, arrowPosY);
            mergeContext.lineTo(currX + arrowLength/2 - arrowheadLength/2, arrowPosY + arrowheadWidth/2);
            mergeContext.lineTo(currX + arrowLength/2 - arrowheadLength/2, arrowPosY + arrowWidth/2);

            // Go back to the left until meeting left arrow head
            mergeContext.lineTo(currX - arrowLength/2 + arrowheadLength/2, arrowPosY + arrowWidth/2);
            
            // Draw left arrow head
            mergeContext.lineTo(currX - arrowLength/2 + arrowheadLength/2, arrowPosY + arrowheadWidth/2);
            mergeContext.lineTo(currX - arrowLength/2, arrowPosY);
            mergeContext.lineTo(currX - arrowLength/2 + arrowheadLength/2, arrowPosY  - arrowheadWidth/2);
            mergeContext.lineTo(currX - arrowLength/2 + arrowheadLength/2, arrowPosY);
            
            mergeContext.lineTo(currX - arrowLength/2 + arrowheadLength/2, arrowPosY - arrowWidth/2);
            mergeContext.lineTo(currX, arrowPosY - arrowWidth/2);

            mergeContext.closePath();

            mergeContext.fillStyle = "#4287f5";
            mergeContext.fill();

            
            
        }
        requestAnimationFrame(drawLoop);
    } 
}

Number.prototype.clamp = function(min, max) {
  return Math.min(Math.max(this, min), max);
};
    
    
function resizeAndPlay(element)
{
  var cv = document.getElementById(element.id + "Merge");
  cv.width = element.videoWidth/2;
  cv.height = element.videoHeight;
  element.play();
  element.style.height = "0px";  // Hide video without stopping it
    
  playVids(element.id);
}


function play(element)
{
    var elements = document.getElementsByClassName(element.id + "-video");
    for(var i  = 0; i< elements.length; i++) 
        if (elements[i].paused){
            elements[i].play();

            if (i == 0){
                element.classList.remove("fa-play");
                element.classList.add("fa-pause");
            }
        }
        else{
            elements[i].pause();

            if (i == 0){
                element.classList.remove("fa-pause");
                element.classList.add("fa-play");
            }
        }
}

function playdata(element)
{

    var datasets = ["aloe", "art", "car", "century", "flowers", "garbage", "picnic", "pikachu", "pipe", "plant", "roses", "table"];
    // pause all videos
    for (var i = 0; i < datasets.length; i++){
        var train_data = document.getElementById(datasets[i] + "-video-train");
        var eval_data = document.getElementById(datasets[i] + "-video-eval");

        if (train_data.paused){}
        else{
            train_data.pause();
            eval_data.pause();
        }
    }

    // start playing the selected videos
    var data = element.id.replace("tab-", "");
    var train_data = document.getElementById(data + "-video-train");
    var eval_data = document.getElementById(data + "-video-eval");

    if (train_data.paused){
        train_data.play();
        eval_data.play();
    }
    else{
        train_data.pause();
        eval_data.pause();
    }
}

function playresults(element)
{
    var datasets = ["aloe", "art", "car", "century", "flowers", "garbage", "picnic", "pikachu", "pipe", "plant", "roses", "table"];

    // pause all videos
    for (var j = 0; j < datasets.length; j++){
        var elements = document.getElementsByClassName(datasets[j] + "-video");
        var button = document.getElementById(datasets[j]);
        for(var i  = 0; i< elements.length; i++) 
            if (elements[i].paused){}
            else{
                elements[i].pause();

                if (i == 0){
                    button.classList.remove("fa-pause");
                    button.classList.add("fa-play");
                }
            }
    }

    // play selected videos
    var results = element.id.replace("tab-", "");
    var elements = document.getElementsByClassName(results + "-video");
    var button = document.getElementById(results);
    for(var i  = 0; i< elements.length; i++) 
        if (elements[i].paused){
            elements[i].play();

            if (i == 0){
                button.classList.remove("fa-play");
                button.classList.add("fa-pause");
            }

        }
}


function updatebaselinetext(element){
    // tab-aloe-nerfacto-vis-sparsity
    var text = element.id.replace("tab-", "");

    var dataset = text.split("-")[0];
    
    var text_element = document.getElementById("baseline-" + dataset);
    text_element.innerHTML = (element.textContent || element.innerText) + " &#8593";

    
}