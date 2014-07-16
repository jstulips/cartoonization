% SIMULATE script for running experiments
%
%
% *** Earlier work by Zoya on pre- and post-processing
% There are three outputs for this framework.
% Output 1: Laplacian High-pass filter + Imadjust(with the range [0.1 , 0.8]) as pre-processing steps
% Output 2: Imadjust (with the range [0.1 , 0.8])as pre-processing step
% Output 3: Imadjust(with the range [0 , 1]) as post-processing step + Unsharp masking as post-processing step

% Read image to process
A = imread('tram.png');
A = im2double(A);

% ***** Earlier code by Zoya for pre-processing steps *********
% *** PRE-PROCESSING ***
%  lapmask=[0 1 0;1 -4 1;0 1 0]; 
%  A = A - imfilter(A ,lapmask,'replicate'); %High pass filter 
%  A = imadjust(A,[0.1 0.8],[]);

options.filtermethod = 'snn';
options.cspace = 'lab';
options.edgemethod = 'dizenzo';
options.nF = 3;           % number of filtering iterations
                          % nF = 4 for BF, nF = 3 for KWH,SNN
options.nE = 2;           % using nE = nF is also OK  
options.nA = 1;           % for SNN only

% Load pre-filtered image to save time (for debugging purposes)
%load bf4_man;
%options.filteredimage = B;
%clear B;     

% *****Earlier code by Zoya for video cartoonization*************
%
%
% workingDir = 'D:\FYP\Video2Frame Conversation\Video8';
% cd(workingDir);
% 
% obj = mmreader('video8.wmv');
% vidFrames = read(obj);
% frames = obj.NumberOfFrames;
% ST='.jpg';
% for x = 1:frames     
%      Sx=num2str(x);
%      Strc=strcat(Sx,ST);
%      VidFrames=vidFrames(:,:,:,x);
%      imwrite(VidFrames,Strc);
% end
% 
% imageNames = dir(fullfile(workingDir,'*.jpg'));
% imageNames = {imageNames.name}';
% imageStrings = regexp([imageNames{:}],'(\d*)','match');
% imageNumbers = str2double(imageStrings);
% [~,sortedIndices] = sort(imageNumbers);
% sortedImageNames = imageNames(sortedIndices);
% outputVideo = VideoWriter(fullfile(workingDir,'cartoon-cl+adj+LHPF.avi'));
% outputVideo.FrameRate = obj.FrameRate;
% open(outputVideo);
% 
% 
% for ii = 1:length(sortedImageNames)
%     A = imread(fullfile(workingDir,sortedImageNames{ii}));
%     A = im2double(A);
%       lapmask=[0 1 0;1 -4 1;0 1 0]; 
%       A = A - imfilter(A ,lapmask,'replicate'); %High pass filter 
%        A = imadjust(A,[0.1 0.8],[]); 
%
%      if ndims(A) == 3      
%
%         r=imadjust(A(:,:,1));
%         g=imadjust(A(:,:,2));
%         b=imadjust(A(:,:,3));
%         A=cat(3,r,g,b);
%      end

Q = cartoonize(A,options);  % cartoonization of each frame
       
% **** POST-PROCESSING (Unsharp masking)      
%         H = fspecial('unsharp');       %Unsharp masking
%        Q = imfilter(Q,H,'replicate');
%        Q=im2uint8(Q);
%     
% 
% 
% 
% writeVideo(outputVideo,Q);
% end
%  
% close(outputVideo);
% 
% 

% [NEW]
% smoothen final output image (for nicer visual look)
% using a 2x2 box filter for minimal effect
Q = imfilter(Q, fspecial('average',[2 2]));
figure; imshow(Q); title('Final stylized output image');

