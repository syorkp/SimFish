classdef (Sealed) EnumeratorBoutInf
    properties (Constant)
    
        
%%%%%%%%%%%%%%%%%%save bout inf%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

fishDirectionX = 1;        
fishDirectionY = 2;    
dataSetNumber = 3;
protocolNumber = 4;
fishUniqueNumber = 5;%updates every fish, desregards split files and counts multitraking fish as independent
errorTailList = 6;%not implemented anymore
stimNumber = 7;
previousStimNumber = 8;
pixelSize = 9;
headToBladderLength = 10;
tailSegmentLength = 11;
distanceEyesToBlob = 12;
lastMeasuredSegment = 13;
fishAge = 14;
boutUniqueNumber = 15;
lagDetector = 16;
numberOfBoutInStim = 17;
uniqueFileNumber = 18;
numberOfBoutAfterCollision = 19;

%%
%%%%%%%%%%%%%%%%%%%%%%%%%% inds %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
indBoutStartAllData = 20;
indBoutEndAllData = 21;

latencyOfBoutAfterStim = 22;

stimUniqueNumber = 23;

%%
%%%%%%%%%%%%%%%%%%%%%%%%%% mistakes %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

brokenBouts = 24;%if bouts have very high ratio

halfBeatPos = 25:1:74;

halfBeatMag = 75:1:124;

doubleBout = 125;

firstSegmentWithMistakes = 126;

firstSegmentWithNotfixedMistakes = 126;
firstSegmentWithAnyMistake  = 127;

%%
%%%%%%%%%%%%%%%%%%%%%%%%%% edges %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

posXStartBout = 128;
posYStartBout = 129;

posXEndBout = 130;
posYEndBout = 131;

%%
%%%%%%%%%%%%%% bout eye conv %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

eyeConv = 132;% 0 no eye conv and 1 eye conv

realFishUniqueNumber = 133;%fish numbers each fish in multitracking is counted and adds split files

boutCat = 134;

rotatedPositionAtSartOfBoutX = 135:1:141;
rotatedPositionAtSartOfBoutY = 142:1:148;

socialBoutsFlag = 149;

coreBout = 150;

mistakesIn74KinSpace = 151;

usedToMakeBehavioralSpace = 152;

trueUniqueFishNumber = 153;%better than fish unique number because it counts multitraking without deviding as one fish

wellNumber = 154;%these only exist in fish with identity
fishIdentityNumber = 155;%these only exist in fish with identity

distToClusterCenter = 156;

eyeConvDiff = 157;
% eyeConvBeforeBout = 158;
% eyeConvAfterBout = 160;

eyeStateCat = 158;

%%
%%%%%%%%%%%%%%%%% new bout cat %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
boutCat2 = 159;

distToClusterCenter2 = 160;


%%
%%%%%%%%%%%%%% strains cat %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
boutCatStrains = 161;%by KNN 11 bout cat

boutCatStrainsGMM = 162;%by GMM 11 bout cat

boutCatStrainsPosterior = 163:175%posterior of 13 GMM

% boutCatStrainsGMM = 162;%by GMM 11 bout cat
% 
% boutCatStrainsPosterior = 163:173%posterior of 11 GMM

    end

    methods (Access = private)    % private so that you cant instantiate
        function out = EnumeratorBoutInf
        end
    end
end