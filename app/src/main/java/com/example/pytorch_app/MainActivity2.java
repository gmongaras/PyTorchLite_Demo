package com.example.pytorch_app;

import androidx.appcompat.app.AppCompatActivity;

import android.graphics.Bitmap;
import android.graphics.Color;
import android.os.Bundle;
import android.util.Log;
import android.view.View;
import android.widget.Button;
import android.widget.ImageView;
import android.widget.TextView;

import org.pytorch.IValue;
import org.pytorch.LiteModuleLoader;
import org.pytorch.Module;
import org.pytorch.Tensor;

import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;
import java.util.Random;

public class MainActivity2 extends AppCompatActivity {

    // Elements in the view
    Button btnGenerate;
    ImageView ivImage;
    TextView tvWaiting;

    // Tag used for logging
    private static final String TAG = "MainActivity2";

    // PyTorch model
    Module module;

    // Size of the input tensor
    int inSize = 512;

    // Width and height of the output image
    int width = 256;
    int height = 256;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main2);

        // Get the elements in the activity
        btnGenerate = findViewById(R.id.btnGenerate);
        ivImage = findViewById(R.id.ivImage);
        tvWaiting = findViewById(R.id.tvWaiting);

        // Load in the model
        try {
            module = LiteModuleLoader.load(assetFilePath("imageGen.pt"));
        } catch (IOException e) {
            Log.e(TAG, "Unable to load model", e);
        }

        // When the button is clicked, generate a new image
        btnGenerate.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View view) {
                // Error handing
                btnGenerate.setClickable(false);
                tvWaiting.setVisibility(View.VISIBLE);

                // Prepare the input tensor. This time, its a
                // a single integer value.
                Tensor inputTensor = generateTensor(inSize);

                // Run the process on a background thread
                new Thread(new Runnable() {
                    @Override
                    public void run() {
                        // Get the output from the model. The
                        // length should be 256*256*3 or 196608
                        float[] outputArr = module.forward(IValue.from(inputTensor)).toTensor().getDataAsFloatArray();

                        // Ensure the output array has values between 0 and 256
                        for (int i = 0; i < outputArr.length; i++) {
                            outputArr[i] = Math.min(Math.max(outputArr[i], 0), 255);
                        }

                        // Create a bitmap of the correct shape
                        Bitmap bmp = Bitmap.createBitmap(width, height, Bitmap.Config.RGB_565);

                        // Iterate over all values in the output tensor
                        // and put them into the bitmap
                        int loc = 0;
                        for (int y = 0; y < width; y++) {
                            for (int x = 0; x < height; x++) {
                                bmp.setPixel(x, y, Color.rgb((int)outputArr[loc], (int)outputArr[loc+1], (int)outputArr[loc+2]));
                                loc += 3;
                            }
                        }

                        // The output of the network is no longer needed
                        outputArr = null;

                        // Resize the bitmap to a larger image
                        bmp = Bitmap.createScaledBitmap(
                                bmp, 512, 512, false);

                        // Display the image
                        Bitmap finalBmp = bmp;
                        runOnUiThread(new Runnable() {
                            @Override
                            public void run() {
                                ivImage.setImageBitmap(finalBmp);

                                // Error handing
                                btnGenerate.setClickable(true);
                                tvWaiting.setVisibility(View.INVISIBLE);
                            }
                        });

                    }
                }).start();

            }
        });
    }

    // Generate a tensor of random doubles given the size of
    // the tensor to generate
    public Tensor generateTensor(int size) {
        // Create a random array of doubles
        Random rand = new Random();
        double[] arr = new double[size];
        for (int i = 0; i < size; i++) {
            arr[i] = -10000 + (double)rand.nextGaussian() * (20000);
        }

        // Create the tensor and return it
        long[] s = {1, size};
        return Tensor.fromBlob(arr, s);
    }


    // Given the name of the pytorch model, get the path for that model
    public String assetFilePath(String assetName) throws IOException {
        File file = new File(this.getFilesDir(), assetName);
        if (file.exists() && file.length() > 0) {
            return file.getAbsolutePath();
        }

        try (InputStream is = this.getAssets().open(assetName)) {
            try (OutputStream os = new FileOutputStream(file)) {
                byte[] buffer = new byte[4 * 1024];
                int read;
                while ((read = is.read(buffer)) != -1) {
                    os.write(buffer, 0, read);
                }
                os.flush();
            }
            return file.getAbsolutePath();
        }
    }
}