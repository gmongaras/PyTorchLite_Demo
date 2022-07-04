package com.example.pytorch_app;

import androidx.appcompat.app.AppCompatActivity;

import android.os.Bundle;
import android.util.Log;
import android.view.View;
import android.widget.Button;
import android.widget.EditText;
import android.widget.TextView;
import android.widget.Toast;

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

public class MainActivity extends AppCompatActivity {

    // Elements in the view
    EditText etNumber;
    Button btnInfer;
    TextView tvDigits;

    // Tag used for logging
    private static final String TAG = "MainActivity";

    // PyTorch model
    Module module;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        // Get all the elements
        etNumber = findViewById(R.id.etNumber);
        btnInfer = findViewById(R.id.btnInfer);
        tvDigits = findViewById(R.id.tvDigits);

        // Load in the model
        try {
            module = LiteModuleLoader.load(assetFilePath("model.pt"));
        } catch (IOException e) {
            Log.e(TAG, "Unable to load model", e);
        }

        // When the button is clicked, generate a noise tensor
        // and get the output from the model
        btnInfer.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View view) {
                // Error checking
                if (etNumber.getText().toString().length() == 0) {
                    Toast.makeText(MainActivity.this, "A number must be supplied", Toast.LENGTH_SHORT).show();
                    return;
                }

                // Get the number of numbers to generate from the edit text
                int N = Integer.parseInt(etNumber.getText().toString());

                // More error checking
                if (N < 1 || N > 10) {
                    Toast.makeText(MainActivity.this, "Digits must be greater than 0 and less than 10", Toast.LENGTH_SHORT).show();
                    return;
                }

                // Prepare the input tensor (N, 2)
                long[] shape = new long[]{N, 2};
                Tensor inputTensor = generateTensor(shape);

                // Get the output from the model
                long[] output = module.forward(IValue.from(inputTensor)).toTensor().getDataAsLongArray();

                // Get the output as a string
                String out = "";
                for (long l : output) {
                    out += String.valueOf(l);
                }

                // Show the output
                tvDigits.setText(out);
            }
        });


    }



    // Generate a tensor of random numbers given the size of that tensor.
    public Tensor generateTensor(long[] Size) {
        // Create a random array of floats
        Random rand = new Random();
        float[] arr = new float[(int)(Size[0]*Size[1])];
        for (int i = 0; i < Size[0]*Size[1]; i++) {
            arr[i] = -10000 + rand.nextFloat() * (20000);
        }

        // Create the tensor and return it
        return Tensor.fromBlob(arr, Size);
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