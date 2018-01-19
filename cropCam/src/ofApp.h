#pragma once

#include "ofMain.h"

class ofApp : public ofBaseApp{

    public:

        void setup();
        void update();
        void draw();

        void keyPressed(int key);
        void keyReleased(int key);
        void mouseMoved(int x, int y);
        void mouseDragged(int x, int y, int button);
        void mousePressed(int x, int y, int button);
        void mouseReleased(int x, int y, int button);
        void mouseEntered(int x, int y);
        void mouseExited(int x, int y);
        void windowResized(int w, int h);
        void dragEvent(ofDragInfo dragInfo);
        void gotMessage(ofMessage msg);
    
        string consoleMessage;
        inline void console(string message){
            consoleMessage = message;
        }

        ofVideoGrabber vidGrabber;
        ofPixels videoInverted;
        ofTexture videoTexture;
        int camWidth;
        int camHeight;
    
        ofImage myImg;
        ofPixels myPix;
    
        ofTrueTypeFont verdana15;
        ofTrueTypeFont verdana30;
};
