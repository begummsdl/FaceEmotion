package com.pena.faceemotion.roomData;


import androidx.room.Dao;
import androidx.room.Delete;
import androidx.room.Insert;
import androidx.room.Query;
import androidx.room.Update;

import java.util.List;

@Dao
public interface IEmotionDAO {
    @Insert
    void insertEmotion(Emotion emotion);
    @Update
    void updateEmotion(Emotion emotion);
    @Delete
    void deleteEmotion(Emotion emotion);

    @Query("SELECT image_path FROM emotions")
    List<String> loadImagePath();

    @Query("SELECT image_path FROM emotions WHERE label_name=:labelName")
    List<String> loadImagePathBylabel(String labelName);

    @Query("SELECT * FROM emotions WHERE image_path=:imagePath")
    List<Emotion> loadEmotionByImagePath(String imagePath);

    @Query("SELECT COUNT(*) FROM emotions WHERE label_name=:labelName AND image_path=:imagePath")
    int pathExists(String labelName,String imagePath);
}
