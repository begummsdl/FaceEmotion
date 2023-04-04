package com.pena.faceemotion;

import android.content.Context;
import android.content.SharedPreferences;
import android.database.Cursor;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.net.Uri;
import android.preference.PreferenceManager;
import android.provider.MediaStore;
import android.util.SparseArray;
import android.widget.Toast;

import com.google.android.gms.vision.Frame;
import com.google.android.gms.vision.face.Face;
import com.google.android.gms.vision.face.FaceDetector;
import com.pena.faceemotion.ml.Mobilenetv2896;
import com.pena.faceemotion.roomData.Emotion;
import com.pena.faceemotion.roomData.EmotionDatabase;
import com.pena.faceemotion.roomData.IEmotionDAO;

import org.tensorflow.lite.DataType;
import org.tensorflow.lite.support.tensorbuffer.TensorBuffer;

import java.io.File;
import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;

public class ImagesGallery {

    public static void listOfImages(Context context) {

        SharedPreferences sharedPreferences = PreferenceManager.getDefaultSharedPreferences(context);
        long lastPhotoDate = sharedPreferences.getLong("last_photo_date", -1);

        String[] projection = { MediaStore.Images.Media.DATA, MediaStore.Images.Media.DATE_ADDED };
        String selection = null;
        String[] selectionArgs = null;
        String sortOrder = MediaStore.Images.Media.DATE_ADDED + " DESC";

        if (lastPhotoDate != -1){
            selection = MediaStore.Images.Media.DATE_ADDED + ">?";
            selectionArgs = new String[] { String.valueOf(lastPhotoDate) };
        }

        Cursor cursor = context.getContentResolver().query(
                MediaStore.Images.Media.EXTERNAL_CONTENT_URI,
                projection,
                selection,
                selectionArgs,
                sortOrder
        );

        int newImageCount = 0;

        if (cursor != null && cursor.moveToFirst()) {
            do {
                String imagePath = cursor.getString(cursor.getColumnIndexOrThrow(MediaStore.Images.Media.DATA));
                long dateTaken = cursor.getLong(cursor.getColumnIndexOrThrow(MediaStore.Images.Media.DATE_ADDED));
                SharedPreferences.Editor editor = sharedPreferences.edit();
                editor.putLong("last_photo_date", dateTaken);
                editor.apply();

                final Bitmap myBitmap = BitmapFactory.decodeFile(imagePath);
                FaceDetector faceDetector = new FaceDetector.Builder(context.getApplicationContext())
                        .setTrackingEnabled(false)
                        .setLandmarkType(FaceDetector.ALL_LANDMARKS)
                        .setMode(FaceDetector.FAST_MODE)
                        .build();
                Frame frame = new Frame.Builder().setBitmap(myBitmap).build();
                SparseArray<Face> sparseArray = faceDetector.detect(frame);

                if (sparseArray.size() > 0){
                    newImageCount++;
                }

                for (int i = 0; i < sparseArray.size(); i++) {
                    Face face = sparseArray.valueAt(i);
                    Bitmap faceBitmap = Bitmap.createBitmap(
                            myBitmap,
                            (int) face.getPosition().x,
                            (int) Math.abs(face.getPosition().y),
                            (int) face.getWidth(),
                            (int) face.getHeight());
                    classifyEmotions(faceBitmap, context, imagePath);
                }
            } while (cursor.moveToNext());
            cursor.close();
            if (newImageCount != 0){
                Toast.makeText(context, newImageCount + " new photo added.", Toast.LENGTH_SHORT).show();
            }
        }
    }

    private static void classifyEmotions(Bitmap imageBitmap,Context context,String ablosutePathOfImage ){
        EmotionDatabase emotionDatabase = EmotionDatabase.getEmotionDatabase(context);
        IEmotionDAO emotionDAO= emotionDatabase.getEmotionDAO();
        try {
            Mobilenetv2896 model = Mobilenetv2896.newInstance(context);
            TensorBuffer inputFeature0 = TensorBuffer.createFixedSize(new int[]{1, 75, 75, 3}, DataType.FLOAT32);

            Bitmap bitmap = Bitmap.createScaledBitmap(imageBitmap, 75,75,false);
            ByteBuffer byteBuffer = ByteBuffer.allocateDirect(4*75*75*3);
            byteBuffer.order(ByteOrder.nativeOrder());

            int[] intValues = new int[75 * 75];
            bitmap.getPixels(intValues, 0, bitmap.getWidth(), 0, 0, bitmap.getWidth(), bitmap.getHeight());

            int pixel = 0;
            for(int i = 0; i < 75; i ++){
                for(int j = 0; j < 75; j++){
                    int val = intValues[pixel++]; // RGB
                    byteBuffer.putFloat(((val >> 16) & 0xFF) * (1.f / 1));
                    byteBuffer.putFloat(((val >> 8) & 0xFF) * (1.f / 1));
                    byteBuffer.putFloat((val & 0xFF) * (1.f / 1));
                }
            }
            inputFeature0.loadBuffer(byteBuffer);
            Mobilenetv2896.Outputs outputs = model.process(inputFeature0);
            TensorBuffer outputFeature0 = outputs.getOutputFeature0AsTensorBuffer();

            float[] confidences = outputFeature0.getFloatArray();
            int maxPos = 0;
            float maxConfidence = 0;
            for (int i = 0; i < confidences.length; i++) {
                if (confidences[i] > maxConfidence) {
                    maxConfidence = confidences[i];
                    maxPos = i;
                }
            }

            String[] classes = {"Angry" ,"Disgust", "Fear" ,"Happy" ,"Sad" ,"Surprise" ,"Neutral"};

            Emotion emotion = new Emotion();
            emotion.setLabelName(classes[maxPos]);
            emotion.setImagePath(ablosutePathOfImage);
            if(emotionDAO.pathExists(emotion.getLabelName(),emotion.getImagePath())!=1) {
                emotionDAO.insertEmotion(emotion);
            }
            model.close();
        } catch (IOException e) {
            // TODO Handle the exception
        }
    }
}