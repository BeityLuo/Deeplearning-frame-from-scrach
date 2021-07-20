#include<cstdio>
#include<cstdlib>

# define TRAIN_IMG "./dataset/train-images.idx3-ubyte"
# define TRAIN_LABEL "./dataset/train-labels.idx1-ubyte"
# define TEST_IMG "./dataset/t10k-images.idx3-ubyte"
# define TEST_LABEL "./dataset/t10k-labels.idx1-ubyte"

# define ROW_NUM   28
# define COLUM_NUM 28

struct int32{
	char data[4];
};

struct Image{
	unsigned char data[ROW_NUM * COLUM_NUM];
};

void show_image(struct Image* img) {
	unsigned char * data = (unsigned char *)img;
	int i = 0, j = 0;
	for (i = 0; i < ROW_NUM; i++) {
		for (j = 0; j < COLUM_NUM; j++) {
			if (*(data + i * ROW_NUM + j) < 128) {
				printf("■ ");
			} else {
				printf("  ");
			}
		}
		printf("\n");
	}
}

int int32Toint(struct int32 num) {
	char *data = &num;
	return data[3] | data[2] << 8 | data[1] << 16 | data[0] << 24;
}
FILE* train_imgs,* train_labels,* test_imgs,* test_labels;
int train_item_num, test_item_num;
void load(){
    train_imgs = fopen(TRAIN_IMG, "r");
    train_labels = fopen(TRAIN_LABEL, "r");
    test_imgs = fopen(TEST_IMG, "r");
    test_labels = fopen(TEST_LABEL, "r");

    // 读取train image文件的开头
    struct int32 magic_num, img_num, row_num, colum_num;
	fread(&magic_num, sizeof(struct int32), 1, train_imgs);
	fread(&img_num, sizeof(struct int32), 1, train_imgs);
	fread(&row_num, sizeof(struct int32), 1, train_imgs);
	fread(&colum_num, sizeof(struct int32), 1, train_imgs);
    train_item_num = int32Toint(img_num);
	// 读取train_label文件的开头
	struct int32 label_num;
	fread(&magic_num, sizeof(struct int32), 1, train_labels);
	fread(&label_num, sizeof(struct int32), 1, train_labels);

	// 读取test image文件的开头
	fread(&magic_num, sizeof(struct int32), 1, test_imgs);
	fread(&img_num, sizeof(struct int32), 1, test_imgs);
	fread(&row_num, sizeof(struct int32), 1, test_imgs);
	fread(&colum_num, sizeof(struct int32), 1, test_imgs);
	test_item_num = int32Toint(img_num);

	// 读取test_label文件的开头
	fread(&magic_num, sizeof(struct int32), 1, test_labels);
	fread(&label_num, sizeof(struct int32), 1, test_labels);
}

int get_train_item_num() {
    return train_item_num;
}

int get_test_item_num() {
    return test_item_num;
}

void next_train_img(char** data) {
    fread(data, ROW_NUM * COLUM_NUM, 1, train_imgs);
}

int next_train_label() {
    char label;
	fread(&label, 1, 1, train_labels);
	return label;
}

void next_test_img(char** data) {
    fread(data, ROW_NUM * COLUM_NUM, 1, test_imgs);
}

int next_test_label() {
    char label;
	fread(&label, 1, 1, test_labels);
	return label;
}

void close(){
    fclose(train_imgs);
    fclose(train_labels);
    fclose(test_imgs);
    fclose(test_labels);
}

void print_test() {
    printf("Hello world!\n");
}

void test();


int main () {
	load();
	char data[28][28];
	next_train_img(data);
	show_image(data);
	close();
}



void test() {
	printf("sizeof(char) = %d, sizeof(unsigned char) = %d\n", sizeof(char), sizeof(unsigned char));
	FILE* train_img = fopen(TRAIN_IMG, "r");
	struct int32 magic_num, img_num, row_num, colum_num;
	int train_label_magic, row, colum;
	
	
	// 读取train image文件的开头
	fread(&magic_num, sizeof(struct int32), 1, train_img);
	printf("train image file: magic_num = %d\n", int32Toint(magic_num));
	fread(&img_num, sizeof(struct int32), 1, train_img);
	fread(&row_num, sizeof(struct int32), 1, train_img);
	fread(&colum_num, sizeof(struct int32), 1, train_img);
	train_label_magic = int32Toint(magic_num);
	row = int32Toint(row_num);
	colum = int32Toint(colum_num);
	printf("train image file: there are %d images in %d * %d shape\n", train_label_magic, row, colum);



	// 读取train_label文件的开头
	FILE* train_label = fopen(TRAIN_LABEL, "r");
	struct int32 label_num;
	fread(&magic_num, sizeof(struct int32), 1, train_label);
	fread(&label_num, sizeof(struct int32), 1, train_label);
	int train_img_magic = int32Toint(magic_num);
	int train_label_num = int32Toint(label_num);
	printf("train label file: magic num is %d, and there are %d labels\n", train_img_magic, train_label_num);

	// 读取一个label
	char label;
	fread(&label, 1, 1, train_label);
	printf("The first num is %d\n", label);
	// 读取train image文件的图像部分
	struct Image* img = (struct Image *)malloc(sizeof(struct Image));
	fread(img, sizeof(struct Image), 1, train_img);
	show_image(img);

	fclose(train_img);
	fclose(train_label);
}
