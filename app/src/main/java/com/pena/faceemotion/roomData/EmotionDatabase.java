package com.pena.faceemotion.roomData;

import android.content.Context;

import androidx.room.Database;
import androidx.room.Room;
import androidx.room.RoomDatabase;

@Database(entities = {Emotion.class}, version = 1, exportSchema= false)
public abstract class EmotionDatabase extends RoomDatabase {
    private static EmotionDatabase emotionDatabase;

    public abstract IEmotionDAO getEmotionDAO();

    private static final String databaseName = "FaceEmotion";

    public static EmotionDatabase getEmotionDatabase(Context context) {
        if (emotionDatabase == null) {
            emotionDatabase =
                    Room.databaseBuilder(context, EmotionDatabase.class, databaseName)
                            .allowMainThreadQueries()
                            .build();
        }
        return emotionDatabase;
    }

    public static void destroyInstance() {
        emotionDatabase = null;
    }
}