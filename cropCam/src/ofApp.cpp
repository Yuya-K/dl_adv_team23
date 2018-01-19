#include "ofApp.h"

//--------------------------------------------------------------
void ofApp::setup(){
    ofTrueTypeFont::setGlobalDpi(72);
    
    camWidth = 640;  // try to grab at this size.
    camHeight = 480;

    //we can now get back a list of devices.
    vector<ofVideoDevice> devices = vidGrabber.listDevices();

    for(int i = 0; i < devices.size(); i++){
        if(devices[i].bAvailable){
            ofLogNotice() << devices[i].id << ": " << devices[i].deviceName;
        }else{
            ofLogNotice() << devices[i].id << ": " << devices[i].deviceName << " - unavailable ";
        }
    }

    vidGrabber.setDeviceID(0);
    vidGrabber.setDesiredFrameRate(60);
    vidGrabber.initGrabber(camWidth, camHeight);

    videoInverted.allocate(camWidth, camHeight, OF_PIXELS_RGB);
    videoTexture.allocate(videoInverted);
    ofSetVerticalSync(true);
    
    //load font
    verdana30.load("./font/verdana.ttf", 30, true, true);
    verdana30.setLineHeight(34.0f);
    verdana30.setLetterSpacing(1.035);
    
    verdana15.load("./font/verdana.ttf", 15, true, true);
    
    myImg.allocate(camWidth, camHeight, OF_IMAGE_COLOR);
}


//--------------------------------------------------------------
void ofApp::update(){
    ofBackground(100, 100, 100);
    vidGrabber.update();
}

//--------------------------------------------------------------
void ofApp::draw(){
    verdana30.drawString("cam image", 20, 80);
    verdana30.drawString("snapped image", 60+camWidth/2, 80);
    ofPushStyle();
    ofSetColor(ofColor(0,50,255));
    verdana15.drawString(consoleMessage, 20, 20);
    ofPopStyle();
    
    ofSetHexColor(0xffffff);
    vidGrabber.draw(20, 100, camWidth/2, camHeight/2);
    myImg.draw(40 + camWidth/2, 100, camWidth/2, camHeight/2);
}


//--------------------------------------------------------------
void ofApp::keyPressed(int key){

    if(key == OF_KEY_RETURN){
        myImg.setFromPixels(vidGrabber.getPixels().getData(),camWidth, camHeight, OF_IMAGE_COLOR);
//        consoleMessage = "image snapped";
        console("image snapped");
    }
    if(key == 's' | key == 'S'){
        myImg.save("./images/output.jpg");
//        consoleMessage = "image saved!";
        console("image saved!");
    }
}

//--------------------------------------------------------------
void ofApp::keyReleased(int key){
}

//--------------------------------------------------------------
void ofApp::mouseMoved(int x, int y){
}

//--------------------------------------------------------------
void ofApp::mouseDragged(int x, int y, int button){
}

//--------------------------------------------------------------
void ofApp::mousePressed(int x, int y, int button){
}

//--------------------------------------------------------------
void ofApp::mouseReleased(int x, int y, int button){
}

//--------------------------------------------------------------
void ofApp::mouseEntered(int x, int y){
}

//--------------------------------------------------------------
void ofApp::mouseExited(int x, int y){
}

//--------------------------------------------------------------
void ofApp::windowResized(int w, int h){
}

//--------------------------------------------------------------
void ofApp::gotMessage(ofMessage msg){
}

//--------------------------------------------------------------
void ofApp::dragEvent(ofDragInfo dragInfo){
}
