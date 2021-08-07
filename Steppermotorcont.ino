/**
 * Library initialization by Teemu Mäntykallio  
 * Serial commands by Daniel Nilsson
 */

#include <TMCStepper.h>
#include <SerialCommands.h>

#define EN_PIN           2 // Enable
#define DIR_PIN          3 // Direction
#define STEP_PIN         5 // Step
#define CS_PIN           4 // Chip select


#define R_SENSE 0.11f // Match to your driver
                      // SilentStepStick series use 0.11
                      // UltiMachine Einsy and Archim2 boards use 0.2
                      // Panucatt BSD2660 uses 0.1
                      // Watterott TMC5160 uses 0.075
float Mem = 0;


TMC2130Stepper driver(CS_PIN, R_SENSE);                           // Hardware SPI


char serial_command_buffer_[32];
void help(SerialCommands* sender);  // Print available commands
void set_rotor_angle(SerialCommands* sender);  // Function to set the angle for the rotor mount
void get_rotor_angle(SerialCommands* sender);  // Function to get the angle from the rotor mount

SerialCommands serial_commands_(&Serial, serial_command_buffer_, sizeof(serial_command_buffer_), "\r\n", " ");
SerialCommand help_("HELP", help);  // Define "HELP" as the serial command
SerialCommand set_rotor_angle_("SRA", set_rotor_angle);  // Define "SRA" as the serial command
SerialCommand get_rotor_angle_("GRA", get_rotor_angle);  // Define "GRA" as the serial comand

void setup() {
  Serial.begin(57600);

  serial_commands_.SetDefaultHandler(cmd_unrecognized);
  serial_commands_.AddCommand(&help_);
  serial_commands_.AddCommand(&set_rotor_angle_);
  serial_commands_.AddCommand(&get_rotor_angle_);
  Serial.println(F("ROTOR MOUNT CONTROLLER: READY!"));

  
  pinMode(EN_PIN, OUTPUT);
  pinMode(STEP_PIN, OUTPUT);
  pinMode(DIR_PIN, OUTPUT);
  digitalWrite(EN_PIN, LOW);      // Enable driver in hardware

  SPI.begin();                    // SPI drivers

  driver.begin();                 //  SPI: Init CS pins and possible SW SPI pins
  
  driver.toff(5);                 // Enables driver in software
  driver.rms_current(1680);        // Set motor RMS current
  driver.microsteps(16);          // Set microsteps to 1/16th

  driver.en_pwm_mode(true);       // Toggle stealthChop on TMC2130/2160/5130/5160
  driver.pwm_autoscale(true);     // Needed for stealthChop
}

bool shaft = false;

void loop() {
  serial_commands_.ReadSerial();
  }


void Move(float deg) {
  for (float i = ((800*deg)/360); i>0; i--) { //800* makes for a full rotation
    digitalWrite(STEP_PIN, HIGH);
    delayMicroseconds(160*7); //*7 makes the motor accelerate and rotate slower allowing for a higher torque
    digitalWrite(STEP_PIN, LOW);
    delayMicroseconds(160*7);
  }
}

void Switchdir(){
  
if(digitalRead(DIR_PIN) == HIGH){
    digitalWrite(DIR_PIN, LOW);
    delay(100);
  }
  else{
    digitalWrite(DIR_PIN, HIGH);
    delay(100);
  }
}

void cmd_unrecognized(SerialCommands* sender, const char* cmd) {  // Default handler 
  sender->GetSerial()->print(F("Unrecognized command: ''"));
  sender->GetSerial()->print(cmd);
  sender->GetSerial()->println(F("'', try ''HELP''!"));
}

void help(SerialCommands* sender) {  // Display help
  sender->GetSerial()->println(F("Commands available:"));
  sender->GetSerial()->println(F("Set Rotor Angle: ''SRA <DEGREES>''"));
  sender->GetSerial()->println(F("Get Rotor Angle: ''GRA'' -> DEGREES"));
}

void set_rotor_angle(SerialCommands* sender) {  // Set the angle [deg] for the rotor mount
  char* arg_str = sender->Next();  //Note: Every call to Next moves the pointer to next parameter
  if (arg_str == NULL) {
    sender->GetSerial()->println(F("ERROR! ROTOR_ANGLE NOT SPECIFIED")); return;
  }
  double rotorAngle = atof(arg_str);  // Convert string to double
  if (Mem < rotorAngle){
    Move(rotorAngle-Mem);
  }
  else if(Mem > rotorAngle){
    Switchdir();
    Move(Mem-rotorAngle);
    Switchdir();
  }
  else{
  }
  Mem = rotorAngle;
}

void get_rotor_angle(SerialCommands* sender) {  // Get the angle [deg] from the rotor mount
  sender->GetSerial()->println(Mem);
}
