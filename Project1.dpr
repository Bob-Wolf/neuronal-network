program Project1;

{$APPTYPE CONSOLE}

{$R *.res}

uses
  System.Threading, System.SysUtils;

type
  TMatrix = array of array of Double;
  TArray = array of Double;
  TFitnessInfo = record
    Fitness: Double;
    Index: Integer;
  end;

var
  Weights:          TMatrix;
  NewWeights:       TMatrix;

  StartValues:      TArray;
  GoalValues:       TArray;
  Biases:           TArray;
  NewBiases:        TArray;
  Fitnesses:        TArray;

  startVal:         Double;
  currentVal:       Double;
  endVal:           Double;
  bestDistance:     Double;
  goalVal:          Double;
  differenceVal:    Double;
  sum:              Double;
  currentInput:     Double;
  goalInput:        Double;

  BestFitnesses:    array of TFitnessInfo;
  tempFitnessInfo:  TFitnessInfo;

  i,j,n,a:          Integer;
  parentAmount:     Integer;
  childIndex:       Integer;
  parentIndex:      Integer;

  taskAmount:       Integer = 35;
  taskSize:         Integer = 10000;
  networkAmount:    Integer = 1000;
  generationAmount: Integer = 400;
  rewardBalance:    Double = 0.01;
  mutationSize:     Double = 0.1;
  iterationAmount:  Integer = 10000;
  threshold:        Double = 0.000005;

begin
  randomize();

  SetLength(Weights, networkAmount, 2);
  SetLength(Biases, networkAmount);
  SetLength(Fitnesses, networkAmount);
  SetLength(BestFitnesses, networkAmount);
  SetLength(NewWeights, networkAmount, 2);
  SetLength(NewBiases, networkAmount);
  SetLength(StartValues, taskAmount);
  SetLength(GoalValues, taskAmount);

  for i := 0 to networkAmount - 1 do
  begin
    Weights[i][0] := (Random * 2.0) - 1.0;
    Weights[i][1] := (Random * 2.0) - 1.0;

    biases[i] := (Random * 2.0) - 1.0;
  end;

  for i := 0 to taskAmount - 1 do
  begin
    StartValues[i] := Random(taskSize) + 1;
    GoalValues[i] := Random(taskSize) + 1;
  end;

  for n := 0 to generationAmount - 1 do
  begin
    if n > 0 then
    begin
      SetLength(Weights, Length(NewWeights));

      for i := 0 to Length(NewWeights) - 1 do
      begin
        Weights[i] := Copy(NewWeights[i], 0, Length(NewWeights[i]));
      end;

      Biases := Copy(NewBiases, 0, Length(NewBiases));
    end;

    { Single Thread
    for i := 0 to networkAmount - 1 do
    begin
      Fitnesses[i] := 0;

      for a := 0 to taskAmount - 1 do
      begin
        currentVal := StartValues[a];
        goalVal := GoalValues[a];
        bestDistance := Abs(goalVal - currentVal);

        for j := 1 to iterationAmount do
        begin
          currentInput := currentVal / taskSize;
          goalInput := goalVal / taskSize;

          sum := biases[i];

          sum := sum + currentInput * Weights[i][0];
          sum := sum + goalInput * Weights[i][1];

          if sum > threshold then
          begin
            currentVal := currentVal + 1
          end
          else if sum < -threshold then
          begin
            currentVal := currentVal - 1
          end;

          differenceVal := Abs(goalVal - currentVal);

          if differenceVal < bestDistance then
          begin
            bestDistance := differenceVal;
          end;
        end;

        endVal := currentVal;

        Fitnesses[i] := Fitnesses[i] + ((Abs(goalVal - endVal)) + rewardBalance * bestDistance);
      end;
    end;
    }

    // Multi Thread
    TParallel.For(0, networkAmount - 1, procedure(i: Integer)
    var
      a, j: Integer;
      localCurrentVal, localGoalVal: Double;
      localBestDistance, localDifferenceVal, localEndVal: Double;
      localCurrentInput, localGoalInput, localSum: Double;
    begin
      Fitnesses[i] := 0;

      for a := 0 to taskAmount - 1 do
      begin
        localCurrentVal := StartValues[a];
        localGoalVal    := GoalValues[a];
        localBestDistance := Abs(localGoalVal - localCurrentVal);

        for j := 1 to iterationAmount do
        begin
          localCurrentInput := localCurrentVal / taskSize;
          localGoalInput    := localGoalVal / taskSize;

          localSum := Biases[i];
          localSum := localSum + localCurrentInput * Weights[i][0];
          localSum := localSum + localGoalInput * Weights[i][1];

          if localSum > threshold then
          begin
            localCurrentVal := localCurrentVal + 1;
          end
          else if localSum < -threshold then
          begin
            localCurrentVal := localCurrentVal - 1;
          end;

          localDifferenceVal := Abs(localGoalVal - localCurrentVal);

          if localDifferenceVal < localBestDistance then
          begin
            localBestDistance := localDifferenceVal;
          end;
        end;

        localEndVal := localCurrentVal;

        Fitnesses[i] := Fitnesses[i] + ((Abs(localGoalVal - localEndVal)) + rewardBalance * localBestDistance);
      end;
    end);

    for i := 0 to networkAmount - 1 do
    begin
      BestFitnesses[i].Fitness := Fitnesses[i];
      BestFitnesses[i].Index := i
    end;

    for i := 0 to networkAmount - 1 - 1 do
    begin
      for j := 0 to networkAmount - 1 - 1 - i do
      begin
        if BestFitnesses[j].Fitness > BestFitnesses[j + 1].Fitness then
        begin
          tempFitnessInfo := BestFitnesses[j];
          BestFitnesses[j] := BestFitnesses[j + 1];
          BestFitnesses[j + 1] := tempFitnessInfo;
        end;
      end;
    end;

    parentAmount := networkAmount div 10;

    if parentAmount < 1 then
      parentAmount := 1;

    if parentAmount > 10 then
      parentAmount := 10;

    for i := 0 to parentAmount - 1 do
    begin
      NewWeights[i][0] :=
        Weights[BestFitnesses[i].Index][0];

      NewWeights[i][1] :=
        Weights[BestFitnesses[i].Index][1];

      NewBiases[i] :=
        Biases[BestFitnesses[i].Index];
    end;

    for childIndex := parentAmount to networkAmount - 1 do
    begin
      parentIndex := (childIndex - parentAmount) mod parentAmount;

      NewWeights[childIndex][0] :=
        NewWeights[parentIndex][0] +
        (Random * (mutationSize * 2)) - mutationSize;

      NewWeights[childIndex][1] :=
        NewWeights[parentIndex][1] +
        (Random * (mutationSize * 2)) - mutationSize;

      NewBiases[childIndex] :=
        NewBiases[parentIndex] +
        (Random * (mutationSize * 2)) - mutationSize;
    end;

    writeln('Generation: ', n, '  Best Fitness: ', BestFitnesses[0].Fitness:0:2);
  end;

  writeln;
  writeln('Finaler Test des besten Netzwerkes');

  startVal := Random(1000000) + 1;
  currentVal := startVal;
  goalVal := Random(1000000) + 1;
  bestDistance := Abs(goalVal - currentVal);

  for j := 1 to 1000000 do
  begin
    currentInput := currentVal / taskSize;
    goalInput := goalVal / taskSize;

    sum := Biases[BestFitnesses[0].Index];

    sum := sum + currentInput * Weights[BestFitnesses[0].Index][0];
    sum := sum + goalInput * Weights[BestFitnesses[0].Index][1];

    if sum > threshold then
    begin
      currentVal := currentVal + 1
    end
    else if sum < -threshold then
    begin
      currentVal := currentVal - 1
    end;

    differenceVal := Abs(goalVal - currentVal);

    if differenceVal < bestDistance then
    begin
    bestDistance := differenceVal;
    end;

    if j = 500000 then
    begin
      writeln;
      writeln('*** ZIEL WIRD GEÄNDERT ***');
      writeln('Altes Goal: ', goalVal:0:0);
      goalVal := goalVal + Round((Random * 80000) - 40000);
      writeln('Neues Goal: ', goalVal:0:0);
    end;
  end;

  endVal := currentVal;
  writeln;
  writeln('Finales Ergebniss');
  writeln('Start: ', startVal:0:2);
  writeln('Goal: ', goalVal:0:2);
  writeln('EndWert: ', endVal:0:2);
  writeln('Fitness: ', ((Abs(goalVal - endVal)) + rewardBalance * bestDistance):0:2);
  writeln;

  readln;
end.
