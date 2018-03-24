#!/usr/bin/env node

'use strict';

const rs2 = require('node-librealsense');
const cv = require('opencv4nodejs');
const pipeline = new rs2.Pipeline();

pipeline.start();

// const rows = 720;
// const cols = 1280;
const winName = 'bob';

function frame() {
    const frameset = pipeline.waitForFrames();
    const depth = frameset.depthFrame;
    if (depth) {
        const matFromArray = new cv.Mat(Buffer.from(depth.data), depth.height, depth.width, cv.CV_8SC1);
        cv.imshow(winName, matFromArray);
        cv.waitKey(1000 / 60);
    }
}

function stop() {
    pipeline.stop();
    pipeline.destroy();
    rs2.cleanup();
    cv.destroyWindow(winName);

    process.exit(0);
}

process.on('exit', stop);
process.on('SIGINT', stop);

setInterval(frame, 1);
