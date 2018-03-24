#!/usr/bin/env node

'use strict';

const rs2 = require('node-librealsense');
const cv = require('opencv4nodejs');

const pipeline = new rs2.Pipeline();

// const config = new rs2.Config();
// config.enableStream(rs2.stream.STREAM_COLOR, -1, 640, 480, rs2.format.FORMAT_RGB8, 30);

pipeline.start(/* config */);

// if (!config.canResolve(pipeline)) {
//     throw new Error('voll dooof!');
// }

const winName = 'bob';

function frame() {
    const frameset = pipeline.waitForFrames();
    const depth = frameset.depthFrame;
    const color = frameset.colorFrame;

    if (depth) {
        const matFromArray = new cv.Mat(Buffer.from(color.data), color.height, color.width, cv.CV_8UC3);
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
