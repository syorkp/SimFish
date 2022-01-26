classdef (Sealed) EnumeratorBoutKinPar
    properties (Constant)
    
        



%%%%%%%%%%%%%%%%%%%%%kinematic parameters%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

numbHalfBeats = 1;
boutDuration = 2;
distBoutDuration = 3;

maxBoutFreq = 4;
minBoutFreq = 5;
meanBoutFreq = 6;

headMaxYaw = 7;
headMeanYaw = 8;
headMinYaw = 9;

boutAngle = 10;
distBoutAngle = 11;
boutMaxAngle = 12;
boutMaxAngularSpeed = 13;
boutAvrAngularSpeed = 14;

boutDistanceX = 15;
boutDistanceY = 16;
boutSpeedX = 17;
boutSpeedY = 18;
distBoutDistanceX = 19;
distBoutDistanceY = 20;
distBoutSpeedX = 21;
distBoutSpeedY = 22;

maxTailChangeAvr = 23;
meanTailChangeAvr = 24;
maxTailChangeMax = 25;
meanTailChangeMax = 26;

boutTailWaveSpeedMax = 27;
boutTailWaveSpeedMin = 28;
boutTailWaveSpeedMean = 29;
boutTailAngularVelocityMax = 30;
boutTailAngularVelocityMin = 31;
boutTailAngularVelocityMean = 32;
boutTailAmplitudePositionRateMax = 33;
boutTailAmplitudePositionRateMin = 34;
boutTailAmplitudePositionRateMean = 35;

boutHalfBendingPosMax = 36;
boutHalfBendingPosMin = 37;
boutHalfBendingPosMean = 38;

boutCruvatureMax = 39;
boutCruvatureMin = 40;
boutCruvatureMean = 41;

boutCurvatureRateMax = 42;
boutCurvatureRateMin = 43;
boutCurvatureRateMean = 44;

boutAbsAmplitude1Avr = 45;
boutAbsAmplitude2Avr = 46;
boutAbsAmplitude3Avr = 47;
boutAbsAmplitude4Avr = 48;
boutAbsAmplitude5Avr = 49;
boutAbsAmplitude6Avr = 50;
boutAbsAmplitude7Avr = 51;
boutAbsAmplitude8Avr = 52;
boutAbsAmplitude9Avr = 53;
boutAbsAmplitude10Avr = 54;

boutAmplitude1Max = 55;
boutAmplitude2Max = 56;
boutAmplitude3Max = 57;
boutAmplitude4Max = 58;
boutAmplitude5Max = 59;
boutAmplitude6Max = 60;
boutAmplitude7Max = 61;
boutAmplitude8Max = 62;
boutAmplitude9Max = 63;
boutAmplitude10Max = 64;

absAUC1 = 65;
absAUC2 = 66;
absAUC3 = 67;
absAUC4 = 68;
absAUC5 = 69;
absAUC6 = 70;
absAUC7 = 71;
absAUC8 = 72;
absAUC9 = 73;
absAUC10 = 74;


%%%%%%%%%1st half beat kin parameters%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
beat1upDown = 75;
beat1numbBeatInBout = 76;
beat1beatDuration = 77;
beat1beatFrequency = 78;

beat1beatAmplitude1= 79;
beat1beatAmplitude2 = 80;
beat1beatAmplitude3 = 81;
beat1beatAmplitude4 = 82; 
beat1beatAmplitude5 = 83;
beat1beatAmplitude6 = 84; 
beat1beatAmplitude7 = 85; 
beat1beatAmplitude8 = 86;
beat1beatAmplitude9 = 87; 
beat1beatAmplitude10 = 88; 
beat1beatAmplitudeAvr = 89;
beat1beatAmplitudeMax = 90;
beat1beatMaxTailAngle = 91;
beat1beatMinTailAngle = 92;
beat1beatHalfBendAngle = 93;
beat1beatMaxTailPos = 94;
beat1beatMinTailPos = 95;
beat1beatHalfBendPos = 96;

beat1beatWaveSpeed = 97;
beat1beatAngularVelocity = 98;
beat1beatAmplitudePositionRate = 99;

beat1AUC1 = 100;
beat1AUC2 = 101;
beat1AUC3 = 102; 
beat1AUC4 = 103;
beat1AUC5 = 104;
beat1AUC6 = 105; 
beat1AUC7 = 106;
beat1AUC8 = 107;
beat1AUC9 = 108;
beat1AUC10 = 109; 
beat1AUCAvr = 110; 
beat1AUCmax = 111;

beat1beatAngle = 112;
beat1beatMaxAngle = 113;
beat1beatMaxAngularSpeed = 114;

beat1beatDistanceX = 115;
beat1beatDistanceY = 116;
beat1beatSpeedX = 117;
beat1beatSpeedY = 118;

beat1maxTailChange = 119;
beat1meanTailChange = 120;

beat1beatMaxCurvature = 121;
beat1beatMaxCurvatureRate = 122;

%%%%%%%%%2nd half beat kin parameters%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
beat2upDown = 123;
beat2numbBeatInBout = 124;
beat2beatDuration = 125;
beat2beatFrequency = 126;

beat2beatAmplitude1= 127;
beat2beatAmplitude2 = 128;
beat2beatAmplitude3 = 129;
beat2beatAmplitude4 = 130; 
beat2beatAmplitude5 = 131;
beat2beatAmplitude6 = 132; 
beat2beatAmplitude7 = 133; 
beat2beatAmplitude8 = 134;
beat2beatAmplitude9 = 135; 
beat2beatAmplitude10 = 136; 
beat2beatAmplitudeAvr = 137;
beat2beatAmplitudeMax = 138;
beat2beatMaxTailAngle = 139;
beat2beatMinTailAngle = 140;

beat2beatHalfBendAngle = 141;
beat2beatMaxTailPos = 142;
beat2beatMinTailPos = 143;
beat2beatHalfBendPos = 144;

beat2beatWaveSpeed = 145;
beat2beatAngularVelocity = 146;
beat2beatAmplitudePositionRate = 147;

beat2AUC1 = 148;
beat2AUC2 = 149;
beat2AUC3 = 150; 
beat2AUC4 = 151;
beat2AUC5 = 152;
beat2AUC6 = 153; 
beat2AUC7 = 154;
beat2AUC8 = 155;
beat2AUC9 = 156;
beat2AUC10 = 157; 
beat2AUCAvr = 158; 
beat2AUCmax = 159;

beat2beatAngle = 160;
beat2beatMaxAngle = 161;
beat2beatMaxAngularSpeed = 162; 

beat2beatDistanceX = 163;
beat2beatDistanceY = 164;
beat2beatSpeedX = 165;
beat2beatSpeedY = 166;
beat2maxTailChange = 167;
beat2meanTailChange = 168;

beat2beatMaxCurvature = 169;
beat2beatMaxCurvatureRate = 170;

%%%%%%%%%3rd half beat kin parameters%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
beat3upDown = 171;
beat3numbBeatInBout = 172;
beat3beatDuration = 173;
beat3beatFrequency = 174;

beat3beatAmplitude1= 175;
beat3beatAmplitude2 = 176;
beat3beatAmplitude3 = 177;
beat3beatAmplitude4 = 178; 
beat3beatAmplitude5 = 179;
beat3beatAmplitude6 = 180; 
beat3beatAmplitude7 = 181; 
beat3beatAmplitude8 = 182;
beat3beatAmplitude9 = 183; 
beat3beatAmplitude10 = 184; 
beat3beatAmplitudeAvr = 185;
beat3beatAmplitudeMax = 186;

beat3beatMaxTailAngle = 187;
beat3beatMinTailAngle = 188;
beat3beatHalfBendAngle = 189;
beat3beatMaxTailPos = 190;
beat3beatMinTailPos = 191;
beat3beatHalfBendPos = 192;

beat3beatWaveSpeed = 193;
beat3beatAngularVelocity = 194;
beat3beatAmplitudePositionRate = 195;

beat3AUC1 = 196;
beat3AUC2 = 197;
beat3AUC3 = 198; 
beat3AUC4 = 199;
beat3AUC5 = 200;
beat3AUC6 = 201; 
beat3AUC7 = 202;
beat3AUC8 = 203;
beat3AUC9 = 204;
beat3AUC10 = 205; 
beat3AUCAvr = 206; 
beat3AUCmax = 207;

beat3beatAngle = 208;
beat3beatMaxAngle = 209;
beat3beatMaxAngularSpeed = 210;

beat3beatDistanceX = 211;
beat3beatDistanceY = 212;
beat3beatSpeedX = 213;
beat3beatSpeedY = 214;
beat3maxTailChange = 215;
beat3meanTailChange = 216;

beat3beatMaxCurvature = 217;
beat3beatMaxCurvatureRate = 218;


firstBeatAmplitudeDiff = 219;
boutAmplitudeDiff = 220;

%%%%%%%%%%%%% corrected bout freq %%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

meanBoutFreqCorr = 221;
maxBoutFreqCorr = 222;
minBoutFreqCorr = 223;


maxDiffBodyAngle = 224;

boutMaxAngleRatio = 225;

%%%%%%%%%%% C1 kin pars %%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
C1Angle = 226;
C1Duration = 227;
C1MaxAngularSpeed = 228;

%%%%%%%%%%%%% C2 kin pars %%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

C2Angle = 229;
C2Duration = 230;
C2MaxAngularSpeed = 231;

%%%%%%%%%%%%% eye conv kin pars %%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

eyeConvAvr = 232;

eyeConv20FramesBeforeBout = 233;
eyeConv20FramesAfterBout = 234;
eyeConvDiff2 = 235;


%%%%%% max angular speeds calcualted with less smooth body angle %%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

boutMaxAngularSpeedLessSmooth = 236;
boutAvrAngularSpeedLessSmooth = 237;

beat1beatAngularVelocitySmooth = 238;
beat2beatAngularVelocitySmooth = 239;
beat3beatAngularVelocitySmooth = 240;

    end

    methods (Access = private)    % private so that you cant instantiate
        function out = EnumeratorBoutKinPar
        end
    end
end