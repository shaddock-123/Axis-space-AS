#define dirPin3 3
#define pulPin3 2

#define dirPin4 4
#define pulPin4 5

#define dirPin5 7
#define pulPin5 6

#define dirPin6 9
#define pulPin6 8

#define dirPin1 10
#define pulPin1 11

#define dirPin2 12
#define pulPin2 13

int a;
void setup() {
  Serial.begin(115200);
  pinMode(pulPin3,OUTPUT); // Step
  pinMode(dirPin3,OUTPUT); // Dir
  pinMode(pulPin4,OUTPUT);
  pinMode(dirPin4,OUTPUT);
  pinMode(pulPin5,OUTPUT);
  pinMode(dirPin5,OUTPUT);
  pinMode(dirPin6,OUTPUT);
  pinMode(pulPin6,OUTPUT);
  pinMode(pulPin1,OUTPUT);
  pinMode(dirPin1,OUTPUT);
  pinMode(pulPin2,OUTPUT);
  pinMode(dirPin2,OUTPUT);
  Serial.begin(115200);
}

void loop() {
  int x=0;
  if(Serial.available()){
    //a = Serial.parseInt();
   a = Serial.read();
    //Serial.print(a);

 
 //-----------------------------------------------------------------------     
    
    if(a == '0'){
      Serial.println("接受成功0");
      //第一节向ZUO弯 
        //测试电机3
      digitalWrite(dirPin2, HIGH); // 使绳紧
      digitalWrite(dirPin5,HIGH);//使绳松
       for (x = 0; x < 15; x++) // Loop 20 times
        {

          digitalWrite(pulPin2, HIGH); // Output high
          digitalWrite(pulPin5,HIGH);

          delayMicroseconds(500); // Wait 1/2 a ms

          digitalWrite(pulPin2, LOW); // Output low
          digitalWrite(pulPin5,LOW);

          delayMicroseconds(500); // Wait 1/2 a ms

         }   
    }
    
    else if(a=='1'){
      Serial.println("接受成功1");
      //第一节向右弯
              //测试电机3
      digitalWrite(dirPin2,LOW); // 使绳紧
      digitalWrite(dirPin5,LOW);//使绳松
       for (x = 0; x < 15; x++) // Loop 200 times
        {

          digitalWrite(pulPin2,HIGH); // Output high
          digitalWrite(pulPin5,HIGH);
          delayMicroseconds(500); // Wait 1/2 a ms

          digitalWrite(pulPin2,LOW); // Output low
          digitalWrite(pulPin5,LOW);
          delayMicroseconds(500); // Wait 1/2 a ms

         }  
      }
      
      
 //-----------------------------------------------------------------------     
      
      else if(a=='2'){
        Serial.println("接受成功2");
        //第一节向上弯
                //测试电机4
        digitalWrite(dirPin1, LOW); // 使绳松
        digitalWrite(dirPin3,LOW);//使绳紧
        for (x = 0; x < 15; x++) // Loop 200 times
         {

          digitalWrite(pulPin1, HIGH); // Output high
          digitalWrite(pulPin3,HIGH);
          delayMicroseconds(500); // Wait 1/2 a ms

          digitalWrite(pulPin1, LOW); // Output low
          digitalWrite(pulPin3,LOW);
          delayMicroseconds(500); // Wait 1/2 a ms

         }  
        }else if(a=='3'){
          Serial.println("接受成功3");
          //第一节向下弯
                  //测试电机4
        digitalWrite(dirPin1, HIGH); // 使绳松
        digitalWrite(dirPin3,HIGH);//使绳紧
        for (x = 0; x < 15; x++) // Loop 200 times
         {

          digitalWrite(pulPin1, HIGH); // Output high
          digitalWrite(pulPin3,HIGH);
          delayMicroseconds(500); // Wait 1/2 a ms

          digitalWrite(pulPin1, LOW); // Output low
          digitalWrite(pulPin3,LOW);
          delayMicroseconds(500); // Wait 1/2 a ms

         }  
          }
          
          
 //-----------------------------------------------------------------------     

 
          else if(a=='4'){
            Serial.println("接受成功4");
            //第二节向YOU弯
                   
            digitalWrite(dirPin6, HIGH); // Set Dir high
            for (x = 0; x < 15; x++) // Loop 200 times
            {

          digitalWrite(pulPin6, HIGH); // Output high

          delayMicroseconds(500); // Wait 1/2 a ms

          digitalWrite(pulPin6, LOW); // Output low

          delayMicroseconds(500); // Wait 1/2 a ms

            }  
            }else if(a=='5'){
              Serial.println("接受成功5");
              //第二节向ZUO弯
                      //测试电机5
              digitalWrite(dirPin6, LOW); // Set Dir high
              for (x = 0; x < 15; x++) // Loop 200 times
              {

                digitalWrite(pulPin6, HIGH); // Output high

                delayMicroseconds(500); // Wait 1/2 a ms

                digitalWrite(pulPin6, LOW); // Output low

                delayMicroseconds(500); // Wait 1/2 a ms

              }  
              }
              
              
              
 //-----------------------------------------------------------------------     
 
              else if(a=='6'){
                Serial.println("接受成功6");
                //第二节向上弯
                        //测试电机6
                digitalWrite(dirPin4, LOW); // Set Dir high
                for (x = 0; x < 15; x++) // Loop 200 times
                {

                   digitalWrite(pulPin4, HIGH); // Output high

                   delayMicroseconds(500); // Wait 1/2 a ms

                   digitalWrite(pulPin4, LOW); // Output low

                   delayMicroseconds(500); // Wait 1/2 a ms

                }  
                }else if(a=='7'){
                  Serial.println("接受成功7");
                  //第二节向下弯
                          //测试电机6
                  digitalWrite(dirPin4, HIGH); // Set Dir high
                   for (x = 0; x < 15; x++) // Loop 200 times
                   {

                   digitalWrite(pulPin4, HIGH); // Output high

                   delayMicroseconds(500); // Wait 1/2 a ms

                   digitalWrite(pulPin4, LOW); // Output low

                   delayMicroseconds(500); // Wait 1/2 a ms

                  }  
                  }
  
  
  
  }
}
