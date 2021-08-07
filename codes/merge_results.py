import os
import sys

# please concat all result files of all parts, and pass the file as argv[1]

EM_round = "#"		# '#' for the last EM epoch. For other epochs, use specific number.
label = "__T__" 	# __T__ for test, __V__ for valid
num_metrics = 4

R = 0

for i in sys.argv[2:]:
	exec(i)

EM_round = str(EM_round)


res = []

with open(sys.argv[1]) as file:
	for line in file:


		if EM_round not in line or label not in line:
			continue

		R += 1

		line = line.split('\n')[0].split('\t')[1:]
		line = [int(line[0])] + list(map(float, line[1:]))
		res.append(line)


res = sorted(res)

print("relations: " + str(R/2))
print("label: " + label)
print("EM_round: " + EM_round)
print("num_metrics: " + str(num_metrics))
assert R % 2 == 0


num_metrics += 1
head = [0] * num_metrics
tail = [0] * num_metrics
total = [0] * num_metrics

for (i, line) in enumerate(res):
	assert line[0] == i
	line = line[1:]

	for (k, arr) in enumerate([head, tail, total]):
		if i < R/2 and k == 1:	continue
		if i >= R/2 and k == 0:	continue

		arr[0] += line[0]

		for i in range(1, num_metrics):
			arr[i] += line[0] * line[i]

for name, arr in zip(["tail", "head", "total"], [tail, head, total]):
	line = [name, "%.0lf" % (arr[0])]
	for i in range(1, num_metrics):
		line.append("%.4lf" % (arr[i] / arr[0]))
	print("\t".join(line))



