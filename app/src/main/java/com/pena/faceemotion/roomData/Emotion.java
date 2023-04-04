package com.pena.faceemotion.roomData;

import androidx.room.ColumnInfo;
import androidx.room.Entity;
import androidx.room.PrimaryKey;

@Entity(tableName = "emotions")
public class Emotion {
    @PrimaryKey(autoGenerate = true)
    @ColumnInfo(name = "image_id")
    private long id;
    @ColumnInfo(name = "label_name")
    private String labelName;
    @ColumnInfo(name = "image_path")
    private String imagePath;

    public long getId() {
        return id;
    }

    public void setId(long id) {
        this.id = id;
    }

    public String getLabelName() {
        return labelName;
    }

    public void setLabelName(String labelName) {
        this.labelName = labelName;
    }

    public String getImagePath() {
        return imagePath;
    }

    public void setImagePath(String imagePath) {
        this.imagePath = imagePath;
    }
}
