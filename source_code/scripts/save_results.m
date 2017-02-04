function [] = save_results()
    
    % Function to save the results
    load('../tmp/ContourPoints.mat','ContourPoints');
    Pnew = InterpolateContourPoints2D(ContourPoints,1000);
    load('../tmp/initial/ContourPoints.mat','ContourPoints');
    I = imread('test.jpg');
    figure(1);
    imshow(I);
    hold on;
    plot(ContourPoints(:,2), ContourPoints(:,1),'b.');
    plot(Pnew(:,2),Pnew(:,1),'r.');
    fig = gcf;
    fig.PaperPositionMode = 'auto';
    legend('Initial Contour','Final Contour');
    title('Deep Active Contour');
    saveas(1,'final_result','jpg');
end
