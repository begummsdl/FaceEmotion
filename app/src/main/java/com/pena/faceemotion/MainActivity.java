package com.pena.faceemotion;

import android.Manifest;
import android.content.ContentValues;
import android.content.Intent;
import android.content.pm.PackageManager;
import android.net.Uri;
import android.os.Bundle;
import android.provider.MediaStore;
import android.view.View;
import android.view.Window;
import android.view.WindowManager;
import android.widget.ImageButton;
import android.widget.TextView;
import android.widget.Toast;

import androidx.annotation.NonNull;
import androidx.appcompat.app.AppCompatActivity;
import androidx.core.app.ActivityCompat;
import androidx.core.content.ContextCompat;
import androidx.core.splashscreen.SplashScreen;
import androidx.recyclerview.widget.RecyclerView;

import com.pena.faceemotion.roomData.EmotionDatabase;
import com.pena.faceemotion.roomData.IEmotionDAO;
import com.pena.faceemotion.utils.MarginDecoration;
import com.pena.faceemotion.utils.PicHolder;
import com.pena.faceemotion.utils.imageFolder;
import com.pena.faceemotion.utils.itemClickListener;
import com.pena.faceemotion.utils.pictureFacer;
import com.pena.faceemotion.utils.pictureFolderAdapter;

import java.util.ArrayList;
import java.util.List;

public class MainActivity extends AppCompatActivity implements itemClickListener {
    private static final int PERMISSION_CODE = 1234;
    private static final int MY_READ_PERMISSION_CODE = 101;
    private static final int CAPTURE_CODE = 1001;
    RecyclerView folderRecycler;
    TextView empty;
    ImageButton cameraBtn;
    Uri image_uri;
    private EmotionDatabase emotionDatabase;
    private IEmotionDAO emotionDAO;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        SplashScreen.installSplashScreen(this);
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        initComponents();
        registerEventHandlers();
    }

    private void initComponents() {
        empty = findViewById(R.id.empty);
        folderRecycler = findViewById(R.id.folderRecycler);
        folderRecycler.addItemDecoration(new MarginDecoration(this));
        folderRecycler.hasFixedSize();
        emotionDatabase = EmotionDatabase.getEmotionDatabase(this);
        emotionDAO = emotionDatabase.getEmotionDAO();
        cameraBtn = findViewById(R.id.cameraBtn);
    }

    private void registerEventHandlers() {
        takePhoto();
        changeStatusBarColor();
    }

    private void changeStatusBarColor() {
        Window window = this.getWindow();
        window.clearFlags(WindowManager.LayoutParams.FLAG_TRANSLUCENT_STATUS);
        window.addFlags(WindowManager.LayoutParams.FLAG_DRAWS_SYSTEM_BAR_BACKGROUNDS);
        window.setStatusBarColor(ContextCompat.getColor(getApplicationContext(), R.color.black));
    }

    private ArrayList<imageFolder> getPicturePaths() {
        ArrayList<imageFolder> picFolders = new ArrayList<>();
        ArrayList<String> picPaths = new ArrayList<>();
        String[] classes = {"Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral"};

        for (int i = 6; i >= 0; i--) {
            for (String x : emotionDAO.loadImagePathBylabel(classes[i])) {
                imageFolder folds = new imageFolder();
                String folder = classes[i];
                List<String> datapaths = emotionDAO.loadImagePathBylabel(classes[i]);
                String datapath = datapaths.get(0);
                if (!picPaths.contains(classes[i])) {
                    picPaths.add(classes[i]);
                    folds.setPath(classes[i]);
                    folds.setFolderName(folder);
                    folds.setFirstPic(datapath);//if the folder has only one picture this line helps to set it as first so as to avoid blank image in itemview
                    folds.addpics();
                    picFolders.add(folds);
                } else {
                    for (int b = 0; b < picFolders.size(); b++) {
                        if (picFolders.get(b).getPath().equals(classes[i])) {
                            picFolders.get(b).setFirstPic(datapath);
                            picFolders.get(b).addpics();
                        }
                    }
                }
            }
        }

        return picFolders;
    }


    private void openCamera() {
        ContentValues values = new ContentValues();
        values.put(MediaStore.Images.Media.TITLE, "new image");
        values.put(MediaStore.Images.Media.DESCRIPTION, "from the camera");
        image_uri = getContentResolver().insert(MediaStore.Images.Media.EXTERNAL_CONTENT_URI, values);
        Intent camIntent = new Intent(MediaStore.ACTION_IMAGE_CAPTURE);
        camIntent.putExtra(MediaStore.EXTRA_OUTPUT, image_uri);
        startActivityForResult(camIntent, CAPTURE_CODE);
    }

    @Override
    protected void onResume() {
        super.onResume();
        loadImages();
        loadFolders();
    }

    private void takePhoto() {
        cameraBtn.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View view) {
                if (checkSelfPermission(Manifest.permission.CAMERA) ==
                        PackageManager.PERMISSION_DENIED ||
                        checkSelfPermission(Manifest.permission.WRITE_EXTERNAL_STORAGE) ==
                                PackageManager.PERMISSION_DENIED) {
                    String[] permission = {Manifest.permission.CAMERA, Manifest.permission.WRITE_EXTERNAL_STORAGE};
                    requestPermissions(permission, PERMISSION_CODE);
                } else {
                    openCamera();
                }
            }
        });
    }
    private void loadImages() {
        if (ContextCompat.checkSelfPermission(MainActivity.this, Manifest.permission.READ_EXTERNAL_STORAGE) != PackageManager.PERMISSION_GRANTED) {
            ActivityCompat.requestPermissions(MainActivity.this, new String[]{Manifest.permission.READ_EXTERNAL_STORAGE}, MY_READ_PERMISSION_CODE);
        } else {
            ImagesGallery.listOfImages(this);
        }
    }

    private void loadFolders() {
        ArrayList<imageFolder> folds = getPicturePaths();
        if (folds.isEmpty()) {
            empty.setVisibility(View.VISIBLE);
        } else {
            RecyclerView.Adapter folderAdapter = new pictureFolderAdapter(folds, MainActivity.this, this);
            folderRecycler.setAdapter(folderAdapter);
        }
    }

    @Override
    public void onPicClicked(PicHolder holder, int position, ArrayList<pictureFacer> pics) {}

    @Override
    public void onPicClicked(String pictureFolderPath, String folderName) {
        Intent move = new Intent(MainActivity.this, ImageDisplay.class);
        move.putExtra("folderPath", pictureFolderPath);
        move.putExtra("folderName", folderName);
        startActivity(move);
    }

    @Override
    public void onRequestPermissionsResult(int requestCode, @NonNull String[] permissions, @NonNull int[] grantResults) {
        super.onRequestPermissionsResult(requestCode, permissions, grantResults);
        switch (requestCode) {
            case PERMISSION_CODE:
                if (grantResults.length > 0 && grantResults[0] == PackageManager.PERMISSION_GRANTED) {
                    openCamera();
                } else {
                    Toast.makeText(this, "Permission denied", Toast.LENGTH_SHORT).show();
                }
            case MY_READ_PERMISSION_CODE:
                if (grantResults[0] == PackageManager.PERMISSION_GRANTED) {
                    Toast.makeText(this, "Read External Storage Permission Granted", Toast.LENGTH_SHORT).show();
                    empty.setVisibility(View.GONE);
                } else {
                    Toast.makeText(this, "Read External Storage Permission Denied", Toast.LENGTH_SHORT).show();
                    empty.setVisibility(View.VISIBLE);
                }
        }
    }
}