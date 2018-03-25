#!/usr/bin/env node

'use strict';

const rs2 = require('node-librealsense');
const cv = require('opencv4nodejs');

/**
 * Render a single frame
 */
function frame({ pipeline }) {
    const frameset = pipeline.waitForFrames();
    // const depth = frameset.depthFrame; // for use if necessary
    const color = frameset.colorFrame;

    if (color) {
        const matFromArray = new cv.Mat(Buffer.from(color.data), color.height, color.width, cv.CV_8UC3);
        cv.imshow('frame', matFromArray);
        cv.waitKey(1000 / 60);
    }
}

/**
 * Stop the application and close all windows
 */
function exit({ pipeline }) {
    pipeline.stop();
    pipeline.destroy();
    rs2.cleanup();
    cv.destroyAllWindows();

    process.exit(0);
}

try {
    const pipeline = new rs2.Pipeline();
    const config = new rs2.Config();

    config.enableStream(rs2.stream.STREAM_COLOR, 0, 640, 480, rs2.format.FORMAT_BGR8, 30);
    pipeline.start(config);

    if (!config.canResolve(pipeline)) {
        throw new Error('Configuration can not be applied to pipeline!');
    }

    process.on('exit', () => exit({ pipeline }));
    process.on('SIGINT', () => exit({ pipeline }));

    setInterval(() => frame({ pipeline }), 1);
} catch (ex) {
    console.error(ex);
}
