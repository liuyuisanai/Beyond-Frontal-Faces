for i = 1 : 100
    imshow(pdata.data_train(:,:,:,i));
    drawnow;
    saveas(gcf,['draw/' num2str(i) '.jpg']);
end