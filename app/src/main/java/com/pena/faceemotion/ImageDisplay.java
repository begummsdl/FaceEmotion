package com.pena.faceemotion;

import android.graphics.BitmapFactory;
import android.os.Bundle;
import android.view.View;
import android.widget.ProgressBar;
import android.widget.TextView;

import androidx.appcompat.app.AppCompatActivity;
import androidx.recyclerview.widget.RecyclerView;

import com.pena.faceemotion.fragments.pictureBrowserFragment;
import com.pena.faceemotion.roomData.EmotionDatabase;
import com.pena.faceemotion.roomData.IEmotionDAO;
import com.pena.faceemotion.utils.MarginDecoration;
import com.pena.faceemotion.utils.PicHolder;
import com.pena.faceemotion.utils.itemClickListener;
import com.pena.faceemotion.utils.pictureFacer;
import com.pena.faceemotion.utils.picture_Adapter;

import java.util.ArrayList;
import java.util.List;


public class ImageDisplay extends AppCompatActivity implements itemClickListener {

    RecyclerView imageRecycler;
    ArrayList<pictureFacer> allpictures;
    ProgressBar load;
    String foldePath;
    TextView folderName;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_image_display);

        folderName = findViewById(R.id.foldername);
        folderName.setText(getIntent().getStringExtra("folderName"));
        foldePath = getIntent().getStringExtra("folderPath");
        allpictures = new ArrayList<>();
        imageRecycler = findViewById(R.id.recycler);
        imageRecycler.addItemDecoration(new MarginDecoration(this));
        imageRecycler.hasFixedSize();
        load = findViewById(R.id.loader);

        if (allpictures.isEmpty()) {
            load.setVisibility(View.VISIBLE);
            allpictures = getAllImagesByFolder(foldePath);
            imageRecycler.setAdapter(new picture_Adapter(allpictures, ImageDisplay.this, this));
            load.setVisibility(View.GONE);
        } else {
        }
    }

    @Override
    public void onPicClicked(PicHolder holder, int position, ArrayList<pictureFacer> pics) {
        pictureBrowserFragment browser = pictureBrowserFragment.newInstance(pics, position, ImageDisplay.this);

        getSupportFragmentManager()
                .beginTransaction()
                .addSharedElement(holder.picture, position + "picture")
                .add(R.id.displayContainer, browser)
                .addToBackStack(null)
                .commit();
    }
    @Override
    public void onPicClicked(String pictureFolderPath, String folderName) {

    }
    public ArrayList<pictureFacer> getAllImagesByFolder(String path) {
        ArrayList<pictureFacer> images = new ArrayList<>();
        EmotionDatabase emotionDatabase = EmotionDatabase.getEmotionDatabase(this);
        IEmotionDAO emotionDAO = emotionDatabase.getEmotionDAO();
        List<String> imagePaths = emotionDAO.loadImagePathBylabel(path);
        for (int i=imagePaths.size()-1; i>=0; i--) {
            String imagePath = imagePaths.get(i);
            pictureFacer pic = new pictureFacer();
            pic.setPicturName(path);
            pic.setPicturePath(imagePath);

            BitmapFactory.Options options = new BitmapFactory.Options();
            options.inJustDecodeBounds = true;
            BitmapFactory.decodeFile(imagePath, options);
            int photoWidth = options.outWidth;
            int photoHeight = options.outHeight;
            Integer boyut = photoWidth * photoHeight;
            pic.setPictureSize(boyut.toString());

            images.add(pic);
        }
        return images;
    }
}